"""Tests for `src/anomaly/attribution.py`.

See AI_CONTEXT.md G4 (per-sensor reconstruction-error attribution) and
Section 10 (reconstruction error) for the contract these tests verify.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.anomaly.attribution import (
    HealthySensorErrorStats,
    WindowAttribution,
    aggregate_sensor_frequency,
    attribute_alerts,
    attribute_alerts_calibrated,
    attribute_window,
    attribute_window_calibrated,
    calibrated_sensor_scores,
    fit_healthy_sensor_error_stats,
    per_sensor_contributions,
    rescale_to_raw_units,
)
from src.data.normalization import RegimeNormalizationStats

FEATURE_COLS = ["setting_1", "setting_2", "setting_3", "sensor_1", "sensor_2", "sensor_3"]
SENSOR_COLS = ["sensor_1", "sensor_2", "sensor_3"]


def test_contributions_sum_to_one_per_window():
    per_channel_error = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        ]
    )
    contributions = per_sensor_contributions(per_channel_error, FEATURE_COLS, SENSOR_COLS)

    assert contributions.shape == (2, 3)
    np.testing.assert_allclose(contributions.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_contributions_exclude_settings_columns():
    # Huge settings error should not affect sensor-only contributions.
    per_channel_error = np.array([[100.0, 100.0, 100.0, 1.0, 1.0, 2.0]])
    contributions = per_sensor_contributions(per_channel_error, FEATURE_COLS, SENSOR_COLS)

    np.testing.assert_allclose(contributions[0], [0.25, 0.25, 0.5], atol=1e-6)


def test_contributions_proportional_to_error():
    per_channel_error = np.array([[0.0, 0.0, 0.0, 1.0, 3.0, 0.0]])
    contributions = per_sensor_contributions(per_channel_error, FEATURE_COLS, SENSOR_COLS)

    np.testing.assert_allclose(contributions[0], [0.25, 0.75, 0.0], atol=1e-6)


def test_zero_error_window_does_not_raise_or_nan():
    per_channel_error = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    contributions = per_sensor_contributions(per_channel_error, FEATURE_COLS, SENSOR_COLS)

    assert not np.isnan(contributions).any()
    np.testing.assert_allclose(contributions[0], [0.0, 0.0, 0.0], atol=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        per_sensor_contributions(np.zeros((2, 6)), FEATURE_COLS[:-1], SENSOR_COLS)


def test_unknown_sensor_col_raises():
    with pytest.raises(ValueError):
        per_sensor_contributions(np.zeros((1, 6)), FEATURE_COLS, ["sensor_99"])


def test_non_2d_error_raises():
    with pytest.raises(ValueError):
        per_sensor_contributions(np.zeros(6), FEATURE_COLS, SENSOR_COLS)


def test_attribute_window_ranks_descending_and_sets_alert():
    row = np.array([0.0, 0.0, 0.0, 1.0, 5.0, 2.0])
    result = attribute_window(
        row, FEATURE_COLS, engine_id=7, end_cycle=120, score=0.9, threshold=0.5, sensor_cols=SENSOR_COLS, top_k=3
    )

    assert isinstance(result, WindowAttribution)
    assert result.engine_id == 7
    assert result.end_cycle == 120
    assert result.alert is True
    assert result.top_sensors == ("sensor_2", "sensor_3", "sensor_1")
    assert result.contributions[0] > result.contributions[1] > result.contributions[2]
    assert sum(result.contributions) == pytest.approx(1.0, abs=1e-6)


def test_attribute_window_below_threshold_is_not_alert():
    row = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    result = attribute_window(row, FEATURE_COLS, engine_id=1, end_cycle=10, score=0.1, threshold=0.5, sensor_cols=SENSOR_COLS)

    assert result.alert is False


def test_attribute_window_top_k_clamped():
    row = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    result = attribute_window(row, FEATURE_COLS, engine_id=1, end_cycle=1, score=1.0, threshold=0.0, sensor_cols=SENSOR_COLS, top_k=10)

    assert len(result.top_sensors) == 3  # clamped to len(sensor_cols)


def test_attribute_window_invalid_top_k_raises():
    row = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        attribute_window(row, FEATURE_COLS, engine_id=1, end_cycle=1, score=1.0, threshold=0.0, top_k=0)


def test_attribute_alerts_only_returns_alerted_windows():
    per_channel_error = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # alert
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # not alert
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # alert
        ]
    )
    engine_ids = np.array([1, 1, 2])
    end_cycles = np.array([10, 11, 50])
    scores = np.array([0.9, 0.1, 0.8])
    alerts = np.array([True, False, True])

    results = attribute_alerts(
        per_channel_error, FEATURE_COLS, engine_ids, end_cycles, scores, 0.5, alerts, sensor_cols=SENSOR_COLS
    )

    assert len(results) == 2
    assert results[0].engine_id == 1
    assert results[0].end_cycle == 10
    assert results[1].engine_id == 2
    assert results[1].end_cycle == 50
    assert all(r.alert for r in results)


def test_attribute_alerts_accepts_per_window_thresholds():
    per_channel_error = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    engine_ids = np.array([1, 2])
    end_cycles = np.array([1, 2])
    scores = np.array([0.6, 0.6])
    thresholds = np.array([0.5, 0.7])  # first alerts, second does not
    alerts = scores > thresholds

    results = attribute_alerts(
        per_channel_error, FEATURE_COLS, engine_ids, end_cycles, scores, thresholds, alerts, sensor_cols=SENSOR_COLS
    )

    assert len(results) == 1
    assert results[0].engine_id == 1


def test_attribute_alerts_length_mismatch_raises():
    per_channel_error = np.zeros((2, 6))
    with pytest.raises(ValueError):
        attribute_alerts(
            per_channel_error,
            FEATURE_COLS,
            engine_ids=np.array([1]),
            end_cycles=np.array([1, 2]),
            scores=np.array([0.1, 0.2]),
            thresholds=0.5,
            alerts=np.array([True, False]),
            sensor_cols=SENSOR_COLS,
        )


def test_aggregate_sensor_frequency_top_rank_only():
    attributions = [
        WindowAttribution(1, 1, 1.0, 0.5, True, ("sensor_2", "sensor_1"), (0.7, 0.3), (7.0, 3.0)),
        WindowAttribution(2, 2, 1.0, 0.5, True, ("sensor_2", "sensor_3"), (0.6, 0.4), (6.0, 4.0)),
        WindowAttribution(3, 3, 1.0, 0.5, True, ("sensor_1", "sensor_2"), (0.55, 0.45), (5.5, 4.5)),
    ]

    freq = aggregate_sensor_frequency(attributions, rank=0)

    assert freq == {"sensor_2": 2, "sensor_1": 1}


def test_aggregate_sensor_frequency_any_rank():
    attributions = [
        WindowAttribution(1, 1, 1.0, 0.5, True, ("sensor_2", "sensor_1"), (0.7, 0.3), (7.0, 3.0)),
        WindowAttribution(2, 2, 1.0, 0.5, True, ("sensor_2", "sensor_3"), (0.6, 0.4), (6.0, 4.0)),
    ]

    freq = aggregate_sensor_frequency(attributions, rank=None)

    assert freq["sensor_2"] == 2
    assert freq["sensor_1"] == 1
    assert freq["sensor_3"] == 1


def test_aggregate_sensor_frequency_empty_input():
    assert aggregate_sensor_frequency([]) == {}


def _make_regime_stats(stds_by_regime: dict[int, list[float]]) -> RegimeNormalizationStats:
    n = len(FEATURE_COLS)
    return RegimeNormalizationStats(
        feature_cols=tuple(FEATURE_COLS),
        regime_col="operating_regime",
        regimes=tuple(sorted(stds_by_regime)),
        means={r: np.zeros(n) for r in stds_by_regime},
        stds={r: np.array(v) for r, v in stds_by_regime.items()},
    )


def test_rescale_to_raw_units_applies_std_squared():
    # std=2.0 for sensor_1 -> raw error = normalized error * 2^2 = *4
    stats = _make_regime_stats({0: [1.0, 1.0, 1.0, 2.0, 0.5, 10.0]})
    per_channel_error = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    regime_ids = np.array([0])

    raw = rescale_to_raw_units(per_channel_error, FEATURE_COLS, regime_ids, stats)

    expected = np.array([[0.0, 0.0, 0.0, 4.0, 0.25, 100.0]])
    np.testing.assert_allclose(raw, expected)


def test_rescale_to_raw_units_near_constant_sensor_shrinks_relative_to_informative_one():
    # sensor_1 near-constant (std=0.01) vs sensor_2 informative (std=5.0),
    # both with the SAME normalized error -- raw rescaling should make
    # sensor_1's contribution shrink drastically relative to sensor_2's,
    # which is exactly the effect this function exists to correct for.
    stats = _make_regime_stats({0: [1.0, 1.0, 1.0, 0.01, 5.0, 1.0]})
    per_channel_error = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])
    regime_ids = np.array([0])

    raw = rescale_to_raw_units(per_channel_error, FEATURE_COLS, regime_ids, stats)
    contributions = per_sensor_contributions(raw, FEATURE_COLS, SENSOR_COLS)

    assert contributions[0, 1] > contributions[0, 0]  # sensor_2 now dominates, not sensor_1


def test_rescale_to_raw_units_per_window_regime():
    stats = _make_regime_stats({0: [1.0, 1.0, 1.0, 2.0, 1.0, 1.0], 1: [1.0, 1.0, 1.0, 10.0, 1.0, 1.0]})
    per_channel_error = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    regime_ids = np.array([0, 1])

    raw = rescale_to_raw_units(per_channel_error, FEATURE_COLS, regime_ids, stats)

    np.testing.assert_allclose(raw[:, 3], [4.0, 100.0])


def test_rescale_to_raw_units_shape_mismatch_raises():
    stats = _make_regime_stats({0: [1.0] * 6})
    with pytest.raises(ValueError):
        rescale_to_raw_units(np.zeros((2, 6)), FEATURE_COLS, np.array([0]), stats)


def test_rescale_to_raw_units_unknown_regime_raises():
    stats = _make_regime_stats({0: [1.0] * 6})
    with pytest.raises(ValueError):
        rescale_to_raw_units(np.zeros((1, 6)), FEATURE_COLS, np.array([7]), stats)


def test_rescale_to_raw_units_missing_feature_col_raises():
    stats = _make_regime_stats({0: [1.0] * 6})
    with pytest.raises(ValueError):
        rescale_to_raw_units(np.zeros((1, 6)), FEATURE_COLS + ["sensor_99"], np.array([0]), stats)


# -- calibrated (per-sensor healthy-baseline) attribution --------------------


def test_fit_healthy_sensor_error_stats_basic():
    healthy_error = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 3.0, 2.0, 1.0],
        ]
    )
    regime_ids = np.array([0, 0])

    stats = fit_healthy_sensor_error_stats(
        healthy_error, FEATURE_COLS, regime_ids, sensor_cols=SENSOR_COLS, min_windows=1
    )

    assert isinstance(stats, HealthySensorErrorStats)
    assert stats.regimes == (0,)
    np.testing.assert_allclose(stats.mean[0], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(stats.std[0], [1.0, 0.0, 1.0])
    assert stats.fallback_regimes == ()


def test_fit_healthy_sensor_error_stats_falls_back_when_too_few_windows():
    healthy_error = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # regime 0
            [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],  # regime 0
            [0.0, 0.0, 0.0, 9.0, 9.0, 9.0],  # regime 1 -- only 1 window
        ]
    )
    regime_ids = np.array([0, 0, 1])

    stats = fit_healthy_sensor_error_stats(
        healthy_error, FEATURE_COLS, regime_ids, sensor_cols=SENSOR_COLS, min_windows=2
    )

    assert stats.fallback_regimes == (1,)
    # regime 1 falls back to the POOLED stats (mean of all 3 windows), not its own single point.
    pooled_mean = healthy_error[:, 3:].mean(axis=0)
    np.testing.assert_allclose(stats.mean[1], pooled_mean)


def test_fit_healthy_sensor_error_stats_all_regimes_covers_unseen_regime():
    # Regime 1 never appears in the calibration data at all (mirrors
    # FD001's val split being dominated by a single KMeans cluster) --
    # with all_regimes passed, it must still get a (pooled-fallback)
    # entry instead of being silently omitted.
    healthy_error = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 3.0, 3.0, 3.0]])
    regime_ids = np.array([0, 0])

    stats = fit_healthy_sensor_error_stats(
        healthy_error,
        FEATURE_COLS,
        regime_ids,
        sensor_cols=SENSOR_COLS,
        min_windows=1,
        all_regimes=range(2),
    )

    assert set(stats.regimes) == {0, 1}
    assert 1 in stats.fallback_regimes
    pooled_mean = healthy_error[:, 3:].mean(axis=0)
    np.testing.assert_allclose(stats.mean[1], pooled_mean)

    # And calibrated_sensor_scores must not raise for that unseen regime.
    test_error = np.array([[0.0, 0.0, 0.0, 5.0, 5.0, 5.0]])
    calibrated = calibrated_sensor_scores(test_error, FEATURE_COLS, np.array([1]), stats)
    assert calibrated.shape == (1, 3)


def test_fit_healthy_sensor_error_stats_empty_raises():
    with pytest.raises(ValueError):
        fit_healthy_sensor_error_stats(np.zeros((0, 6)), FEATURE_COLS, np.array([]), sensor_cols=SENSOR_COLS)


def test_fit_healthy_sensor_error_stats_invalid_min_windows_raises():
    with pytest.raises(ValueError):
        fit_healthy_sensor_error_stats(
            np.zeros((2, 6)), FEATURE_COLS, np.array([0, 0]), sensor_cols=SENSOR_COLS, min_windows=0
        )


def test_calibrated_scores_zero_at_healthy_mean():
    healthy_error = np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 3.0, 2.0, 1.0]])
    stats = fit_healthy_sensor_error_stats(
        healthy_error, FEATURE_COLS, np.array([0, 0]), sensor_cols=SENSOR_COLS, min_windows=1
    )

    # A window whose error exactly equals the healthy mean should calibrate to ~0.
    at_mean = np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]])
    calibrated = calibrated_sensor_scores(at_mean, FEATURE_COLS, np.array([0]), stats)

    np.testing.assert_allclose(calibrated, [[0.0, 0.0, 0.0]], atol=1e-4)


def test_calibrated_scores_positive_when_above_healthy_baseline():
    healthy_error = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]] * 5)
    stats = fit_healthy_sensor_error_stats(
        healthy_error, FEATURE_COLS, np.array([0] * 5), sensor_cols=SENSOR_COLS, min_windows=1
    )

    elevated = np.array([[0.0, 0.0, 0.0, 10.0, 1.0, 1.0]])
    calibrated = calibrated_sensor_scores(elevated, FEATURE_COLS, np.array([0]), stats)

    assert calibrated[0, 0] > calibrated[0, 1]
    assert calibrated[0, 0] > calibrated[0, 2]


def test_calibrated_scores_invariant_to_uniform_rescaling():
    # Simulates the raw-vs-normalized invariance claimed in the module
    # docstring: scaling healthy AND test error by the same per-sensor
    # constant (e.g. std_j^2, as in rescale_to_raw_units) must not
    # change the calibrated z-score.
    healthy_error = np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 3.0, 4.0, 1.0], [0.0, 0.0, 0.0, 2.0, 1.0, 5.0]])
    test_error = np.array([[0.0, 0.0, 0.0, 8.0, 2.0, 1.0]])
    regime_ids_healthy = np.array([0, 0, 0])
    regime_ids_test = np.array([0])

    stats_a = fit_healthy_sensor_error_stats(
        healthy_error, FEATURE_COLS, regime_ids_healthy, sensor_cols=SENSOR_COLS, min_windows=1, eps=0.0
    )
    calibrated_a = calibrated_sensor_scores(test_error, FEATURE_COLS, regime_ids_test, stats_a)

    scale = np.array([4.0, 0.25, 9.0])  # arbitrary per-sensor constant
    healthy_scaled = healthy_error.copy()
    healthy_scaled[:, 3:] *= scale
    test_scaled = test_error.copy()
    test_scaled[:, 3:] *= scale

    stats_b = fit_healthy_sensor_error_stats(
        healthy_scaled, FEATURE_COLS, regime_ids_healthy, sensor_cols=SENSOR_COLS, min_windows=1, eps=0.0
    )
    calibrated_b = calibrated_sensor_scores(test_scaled, FEATURE_COLS, regime_ids_test, stats_b)

    np.testing.assert_allclose(calibrated_a, calibrated_b, atol=1e-6)


def test_calibrated_scores_unknown_regime_raises():
    healthy_error = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    stats = fit_healthy_sensor_error_stats(
        healthy_error, FEATURE_COLS, np.array([0]), sensor_cols=SENSOR_COLS, min_windows=1
    )
    with pytest.raises(ValueError):
        calibrated_sensor_scores(np.zeros((1, 6)), FEATURE_COLS, np.array([7]), stats)


def test_attribute_window_calibrated_ranks_by_zscore_not_magnitude():
    # sensor_1 usually errors around 100 (big, but typical for it);
    # sensor_2 usually errors around 1 but this window is way above
    # that -- calibrated ranking should put sensor_2 first even though
    # its RAW error is smaller.
    healthy_error = np.array([[0.0, 0.0, 0.0, 100.0, 1.0, 1.0]] * 5)
    stats = fit_healthy_sensor_error_stats(
        healthy_error, FEATURE_COLS, np.array([0] * 5), sensor_cols=SENSOR_COLS, min_windows=1
    )

    window_error = np.array([0.0, 0.0, 0.0, 105.0, 10.0, 1.0])  # sensor_2 way above its tiny baseline

    result = attribute_window_calibrated(
        window_error, FEATURE_COLS, regime_id=0, stats=stats, engine_id=1, end_cycle=1, score=1.0, threshold=0.5, top_k=3
    )

    assert result.top_sensors[0] == "sensor_2"


def test_attribute_alerts_calibrated_only_returns_alerted_windows():
    healthy_error = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]] * 5)
    stats = fit_healthy_sensor_error_stats(
        healthy_error, FEATURE_COLS, np.array([0] * 5), sensor_cols=SENSOR_COLS, min_windows=1
    )

    per_channel_error = np.array(
        [
            [0.0, 0.0, 0.0, 10.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 10.0, 1.0],
        ]
    )
    regime_ids = np.array([0, 0])
    engine_ids = np.array([1, 2])
    end_cycles = np.array([10, 20])
    scores = np.array([0.9, 0.1])
    alerts = np.array([True, False])

    results = attribute_alerts_calibrated(
        per_channel_error, FEATURE_COLS, regime_ids, stats, engine_ids, end_cycles, scores, 0.5, alerts
    )

    assert len(results) == 1
    assert results[0].engine_id == 1
    assert results[0].top_sensors[0] == "sensor_1"


def test_attribute_alerts_calibrated_length_mismatch_raises():
    healthy_error = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    stats = fit_healthy_sensor_error_stats(
        healthy_error, FEATURE_COLS, np.array([0]), sensor_cols=SENSOR_COLS, min_windows=1
    )
    with pytest.raises(ValueError):
        attribute_alerts_calibrated(
            np.zeros((2, 6)),
            FEATURE_COLS,
            regime_ids=np.array([0, 0]),
            stats=stats,
            engine_ids=np.array([1]),
            end_cycles=np.array([1, 2]),
            scores=np.array([0.1, 0.2]),
            thresholds=0.5,
            alerts=np.array([True, False]),
        )
