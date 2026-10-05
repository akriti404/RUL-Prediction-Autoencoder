"""Tests for `src/anomaly/attribution.py`.

See AI_CONTEXT.md G4 (per-sensor reconstruction-error attribution) and
Section 10 (reconstruction error) for the contract these tests verify.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.anomaly.attribution import (
    WindowAttribution,
    aggregate_sensor_frequency,
    attribute_alerts,
    attribute_window,
    per_sensor_contributions,
)

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
