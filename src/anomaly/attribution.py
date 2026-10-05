"""Per-sensor reconstruction-error attribution and ranking.

See AI_CONTEXT.md Section 19 (`anomaly/attribution.py` module
contract), G4 (per-sensor reconstruction-error attribution), and
Section 10 (reconstruction error) -- this module consumes the
channel-wise error `anomaly/reconstruction.py` deliberately preserves
(`per_channel_error`, E_j) rather than re-deriving it from the already
channel-collapsed `window_score`.

Design decisions:

  - Attribution is computed over SENSOR channels only, excluding the
    3 operating settings (AI_CONTEXT.md Section 3.3: "the model should
    not blindly treat all 24 variables identically"). G4 is about
    explaining *sensor* behavior; the settings are regime-conditioning
    inputs, not channels whose reconstruction error indicates
    degradation.
  - Contribution is a per-window fraction, contribution_j = E_j / sum_j
    E_j (restricted to the sensor subset), so contributions are
    comparable across windows/engines/experiments regardless of each
    window's absolute error scale, and sum to 1 across sensors.
  - Attribution is reported for ALERTED windows only (`attribute_alerts`),
    since the G4 output ("alert should contain ... top contributing
    sensors") explains an alert, not every healthy window. Lower-level
    helpers (`per_sensor_contributions`, `attribute_window`) operate on
    any window and are reused by `attribute_alerts`.

RAW-UNIT RESCALING (added after diagnosing the normalized-score
ranking, Oct 2026): the model operates on per-regime z-score
normalized features, so `per_channel_error` computed directly from
model input/output is in NORMALIZED units -- E_j / sum_j E_j therefore
answers "which sensor's error is large relative to ITS OWN regime
variance," not "which sensor's error is large in absolute physical
terms." A sensor with tiny natural within-regime variance (near-
constant, e.g. a sensor sitting at std ~0.003 after per-regime
z-scoring) gets any small absolute reconstruction noise divided by
that tiny std, inflating its normalized error and making it dominate
the ranking despite not being an informative degradation channel --
confirmed on FD002, where `sensor_6` (within-regime std ~0.0034, far
below sensors generally considered informative in C-MAPSS literature)
topped the normalized-score ranking.

`rescale_to_raw_units` converts `per_channel_error` back to raw
(physical sensor-unit) squared error, so `attribute_alerts` can also
be run in raw units as a cross-check against the normalized ranking --
see `run_attribution.py`, which reports both.

PER-SENSOR CALIBRATION (added after the raw-unit cross-check revealed
a MIRROR-IMAGE bias, Oct 2026): raw units fixed the near-constant-
sensor inflation, but introduced the opposite problem -- raw_E_j =
std_j^2 * normalized_E_j means sensors with the LARGEST absolute
physical variance (regardless of reconstruction quality) mechanically
dominate the raw ranking, since the autoencoder's normalized errors
are roughly comparable in scale across sensors by construction
(that's the whole point of training on normalized features). Confirmed
on FD002: raw-unit attribution collapsed onto exactly the 4
highest-variance sensors (sensor_3/4/9/14) for essentially every
alert, with the other 17 sensors never winning once.

Neither the normalized nor the raw view is therefore trustworthy
alone -- one is biased toward low-variance sensors, the other toward
high-variance ones. `fit_healthy_sensor_error_stats` /
`calibrated_sensor_scores` fix this by comparing each sensor's error
ONLY AGAINST ITS OWN typical (healthy-region) reconstruction error,
per regime:

    calibrated_j = (E_j(window) - mean_healthy_j(regime)) / std_healthy_j(regime)

This is a per-channel z-score of the ERROR itself (not of the raw
feature), mirroring the variance-correction already validated for the
scalar window score (`reconstruction.normalized_anomaly_scores`) --
just applied per sensor instead of to the pooled total. Because
raw_E_j = std_j(regime)^2 * normalized_E_j with the SAME std_j for
both a window and its healthy baseline (same regime), the std_j^2
factor cancels in the z-score: `calibrated_j` comes out IDENTICAL
whether computed from normalized or raw per-channel error, which is
exactly why it is immune to the scale bias affecting both of those
views on their own -- it asks "is this sensor reconstructing worse
than USUAL for it," never "is this sensor's error big compared to
some OTHER sensor's natural scale."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from src.data.schema import SENSOR_COLUMNS

if TYPE_CHECKING:
    from src.data.normalization import RegimeNormalizationStats


@dataclass(frozen=True)
class WindowAttribution:
    """Per-sensor attribution/explanation for a single window.

    Mirrors the alert-object fields AI_CONTEXT.md Section G4 asks for:
    "total anomaly score, threshold, alert status, top contributing
    sensors, contribution values, engine ID, cycle/window endpoint."

    Attributes:
        engine_id: Engine this window belongs to.
        end_cycle: The window's "as-of" (last-row) cycle.
        score: The window's total anomaly score (e.g.
            `normalized_anomaly_scores`).
        threshold: The threshold `score` was compared against for this
            window (may vary per window under a regime-aware/conformal
            threshold).
        alert: Whether this window was alerted (score > threshold).
        top_sensors: Sensor names, ranked by contribution descending
            (most-contributing first).
        contributions: Contribution fraction for each entry in
            `top_sensors`, same order, in [0, 1].
        raw_errors: The underlying per-channel error (E_j) for each
            entry in `top_sensors`, same order -- useful for inspecting
            absolute magnitude alongside the normalized fraction.
    """

    engine_id: object
    end_cycle: int
    score: float
    threshold: float
    alert: bool
    top_sensors: tuple[str, ...]
    contributions: tuple[float, ...]
    raw_errors: tuple[float, ...]


def per_sensor_contributions(
    per_channel_error: np.ndarray,
    feature_cols: Sequence[str],
    sensor_cols: Sequence[str] = SENSOR_COLUMNS,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize per-channel error into per-window sensor contribution fractions.

    contribution_j = E_j / sum_{j in sensor_cols} E_j

    Args:
        per_channel_error: Shape (n_windows, n_features), e.g.
            `anomaly.reconstruction.per_channel_error(...).numpy()`.
        feature_cols: Column names for the `n_features` axis, in the
            SAME order `per_channel_error` was computed with (e.g. the
            checkpoint's `feature_cols`).
        sensor_cols: Subset of `feature_cols` to attribute over.
            Defaults to all 21 C-MAPSS sensors (excludes operating
            settings -- see module docstring).
        eps: Numerical floor added to each window's sensor-error sum,
            so a window with (near-)zero total sensor error produces
            (near-)zero contributions rather than raising/NaN-ing.

    Returns:
        Array of shape (n_windows, len(sensor_cols)): contribution
        fraction per sensor, per window. Rows sum to ~1 (exactly 1 when
        `eps` is negligible relative to the row's error sum).

    Raises:
        ValueError: If `per_channel_error` is not 2-D, its second
            dimension doesn't match `len(feature_cols)`, or any column
            in `sensor_cols` is not present in `feature_cols`.
    """
    per_channel_error = np.asarray(per_channel_error, dtype=float)
    feature_cols = list(feature_cols)
    sensor_cols = list(sensor_cols)

    if per_channel_error.ndim != 2:
        raise ValueError(
            f"per_channel_error must be 2-D (n_windows, n_features), got shape {per_channel_error.shape}"
        )
    if per_channel_error.shape[1] != len(feature_cols):
        raise ValueError(
            f"per_channel_error has {per_channel_error.shape[1]} feature columns, "
            f"but feature_cols has {len(feature_cols)}"
        )
    missing = set(sensor_cols) - set(feature_cols)
    if missing:
        raise ValueError(f"sensor_cols not present in feature_cols: {sorted(missing)}")

    sensor_idx = [feature_cols.index(col) for col in sensor_cols]
    sensor_error = per_channel_error[:, sensor_idx]
    row_sums = sensor_error.sum(axis=1, keepdims=True)
    return sensor_error / (row_sums + eps)


def rescale_to_raw_units(
    per_channel_error: np.ndarray,
    feature_cols: Sequence[str],
    regime_ids: np.ndarray,
    normalization_stats: "RegimeNormalizationStats",
) -> np.ndarray:
    """Rescale per-channel error from normalized (z-score) units to raw sensor units.

    Since each feature is normalized as `z = (raw - mean) / std`, the
    reconstruction difference in raw units is `std * (z - z_hat)`, so
    the SQUARED raw-unit error is `std^2` times the squared normalized
    error. Because `per_channel_error` (E_j, see
    `anomaly/reconstruction.py`) is itself an average of squared
    per-timestep errors, and `std_j` is constant across a window's
    timesteps (see caveat below), this reduces to an exact per-window,
    per-channel scalar rescale:

        raw_E_j = std_j(regime)^2 * normalized_E_j

    Args:
        per_channel_error: Shape (n_windows, n_features), normalized-
            unit per-channel error (e.g. from
            `anomaly.reconstruction.per_channel_error`).
        feature_cols: Column names for the `n_features` axis, in order.
        regime_ids: The regime assigned to EACH WINDOW (one regime per
            window, e.g. from the window's last-cycle settings -- the
            same convention `run_regime_conditioned.py` /
            `run_conformal.py` use for model conditioning and threshold
            lookup), shape (n_windows,).
        normalization_stats: The `RegimeNormalizationStats` the
            checkpoint was trained with (`checkpoint["normalization_stats"]`),
            giving each regime's fitted std per feature.

    Returns:
        Array of shape (n_windows, n_features): per-channel error in
        raw (physical) sensor units.

    Raises:
        ValueError: If shapes mismatch, `feature_cols` isn't a subset
            of `normalization_stats.feature_cols`, or a window's
            `regime_ids` value has no entry in `normalization_stats.stds`.

    Caveat: this treats EVERY row of a window as belonging to that
    window's single assigned regime (its std is applied uniformly
    across the whole window). If the true operating regime changed
    mid-window, this is an approximation -- but it is the SAME
    single-regime-per-window approximation already used elsewhere in
    this pipeline (model conditioning, conformal threshold lookup), so
    staying consistent with it here rather than introducing a
    different assumption.
    """
    per_channel_error = np.asarray(per_channel_error, dtype=float)
    feature_cols = list(feature_cols)
    regime_ids = np.asarray(regime_ids)

    if per_channel_error.ndim != 2:
        raise ValueError(
            f"per_channel_error must be 2-D (n_windows, n_features), got shape {per_channel_error.shape}"
        )
    if per_channel_error.shape[1] != len(feature_cols):
        raise ValueError(
            f"per_channel_error has {per_channel_error.shape[1]} feature columns, "
            f"but feature_cols has {len(feature_cols)}"
        )
    if per_channel_error.shape[0] != len(regime_ids):
        raise ValueError(
            f"per_channel_error and regime_ids must have the same number of windows, "
            f"got {per_channel_error.shape[0]} vs {len(regime_ids)}"
        )

    stats_cols = list(normalization_stats.feature_cols)
    missing = set(feature_cols) - set(stats_cols)
    if missing:
        raise ValueError(f"feature_cols not present in normalization_stats: {sorted(missing)}")
    reorder = [stats_cols.index(col) for col in feature_cols]

    raw_error = np.empty_like(per_channel_error)
    for regime in np.unique(regime_ids):
        regime_int = int(regime)
        if regime_int not in normalization_stats.stds:
            raise ValueError(f"No normalization stats for regime {regime_int}")
        mask = regime_ids == regime
        std = np.asarray(normalization_stats.stds[regime_int])[reorder]
        raw_error[mask] = per_channel_error[mask] * (std**2)

    return raw_error


@dataclass(frozen=True)
class HealthySensorErrorStats:
    """Per-regime, per-sensor baseline reconstruction-error statistics.

    Fitted on HEALTHY calibration windows only (AI_CONTEXT.md Section
    17 Rule 3/4 -- never on test data, never on windows also used to
    fit the model itself). One mean/std per (regime, sensor) pair.

    Attributes:
        sensor_cols: Sensor column order `mean`/`std` are indexed by.
        regimes: Every regime this was fit for.
        mean, std: Per-regime arrays, each shape (len(sensor_cols),) --
            that regime's healthy mean/std of `per_channel_error` for
            each sensor.
        fallback_regimes: Regimes whose stats fell back to the POOLED
            (all-regime) healthy stats because fewer than `min_windows`
            healthy calibration windows were available for that regime
            specifically (mirrors `anomaly.conformal`'s per-regime
            fallback pattern, for the same reason: too few points makes
            a per-regime estimate unreliable).
        eps: Floor added to `std` before dividing, so a sensor with
            (near-)zero healthy-region error variance doesn't produce
            an exploding/undefined z-score.
    """

    sensor_cols: tuple[str, ...]
    regimes: tuple[int, ...]
    mean: dict[int, np.ndarray]
    std: dict[int, np.ndarray]
    fallback_regimes: tuple[int, ...]
    eps: float


def fit_healthy_sensor_error_stats(
    healthy_per_channel_error: np.ndarray,
    feature_cols: Sequence[str],
    regime_ids: np.ndarray,
    sensor_cols: Sequence[str] = SENSOR_COLUMNS,
    min_windows: int = 30,
    eps: float = 1e-8,
    all_regimes: Sequence[int] | None = None,
) -> HealthySensorErrorStats:
    """Fit each sensor's typical (healthy) reconstruction-error mean/std, per regime.

    Args:
        healthy_per_channel_error: Shape (n_windows, n_features),
            `per_channel_error` computed on HEALTHY calibration
            (val-split) windows only -- e.g. the same val windows
            `run_conformal.py` uses to fit the conformal threshold.
        feature_cols: Column names for the `n_features` axis, in order.
        regime_ids: The regime assigned to each calibration window
            (same convention as `rescale_to_raw_units`), shape
            (n_windows,).
        sensor_cols: Subset of `feature_cols` to fit stats for.
        min_windows: Minimum healthy windows a regime must have before
            it gets its OWN mean/std; below this, falls back to the
            pooled (all-regime) stats, since a per-regime estimate from
            very few windows is unreliable.
        eps: See `HealthySensorErrorStats.eps`.
        all_regimes: The FULL set of regime IDs the regime model can
            ever produce (e.g. `range(regime_model.n_regimes)`), not
            just the regimes that happen to appear in this calibration
            set. Pass this explicitly whenever the calibration set may
            not cover every regime (e.g. FD001's val split, which can
            be dominated by a single KMeans cluster even though test
            windows later get assigned others) -- mirrors
            `anomaly.conformal.fit_conformal_thresholds_per_regime`'s
            `all_regimes` parameter and fixes the exact same class of
            bug: a regime with ZERO calibration windows would otherwise
            get no stats at all and raise in `calibrated_sensor_scores`
            the first time it's seen at test time, instead of falling
            back to the pooled baseline like an under-`min_windows`
            regime does. If None (default), only regimes observed in
            `regime_ids` get stats.

    Returns:
        A `HealthySensorErrorStats` covering every regime in
        `all_regimes` (or, if not given, every regime present in
        `regime_ids`).

    Raises:
        ValueError: If shapes mismatch, `sensor_cols` isn't a subset of
            `feature_cols`, `healthy_per_channel_error` is empty, or
            `min_windows` is not a positive integer.
    """
    healthy_per_channel_error = np.asarray(healthy_per_channel_error, dtype=float)
    feature_cols = list(feature_cols)
    sensor_cols = list(sensor_cols)
    regime_ids = np.asarray(regime_ids)

    if healthy_per_channel_error.ndim != 2:
        raise ValueError(
            f"healthy_per_channel_error must be 2-D, got shape {healthy_per_channel_error.shape}"
        )
    if healthy_per_channel_error.shape[0] == 0:
        raise ValueError("healthy_per_channel_error must be non-empty")
    if healthy_per_channel_error.shape[1] != len(feature_cols):
        raise ValueError(
            f"healthy_per_channel_error has {healthy_per_channel_error.shape[1]} feature columns, "
            f"but feature_cols has {len(feature_cols)}"
        )
    if healthy_per_channel_error.shape[0] != len(regime_ids):
        raise ValueError(
            "healthy_per_channel_error and regime_ids must have the same number of windows, "
            f"got {healthy_per_channel_error.shape[0]} vs {len(regime_ids)}"
        )
    missing = set(sensor_cols) - set(feature_cols)
    if missing:
        raise ValueError(f"sensor_cols not present in feature_cols: {sorted(missing)}")
    if not isinstance(min_windows, int) or min_windows <= 0:
        raise ValueError(f"min_windows must be a positive integer, got {min_windows!r}")

    sensor_idx = [feature_cols.index(col) for col in sensor_cols]
    sensor_error = healthy_per_channel_error[:, sensor_idx]

    pooled_mean = sensor_error.mean(axis=0)
    pooled_std = sensor_error.std(axis=0)

    regimes = (
        tuple(sorted(set(int(r) for r in all_regimes) | set(int(r) for r in np.unique(regime_ids))))
        if all_regimes is not None
        else tuple(sorted(int(r) for r in np.unique(regime_ids)))
    )
    mean: dict[int, np.ndarray] = {}
    std: dict[int, np.ndarray] = {}
    fallback_regimes: list[int] = []

    for regime in regimes:
        mask = regime_ids == regime
        n_windows = int(mask.sum())
        if n_windows < min_windows:
            mean[regime] = pooled_mean
            std[regime] = pooled_std
            fallback_regimes.append(regime)
            continue
        mean[regime] = sensor_error[mask].mean(axis=0)
        std[regime] = sensor_error[mask].std(axis=0)

    return HealthySensorErrorStats(
        sensor_cols=tuple(sensor_cols),
        regimes=regimes,
        mean=mean,
        std=std,
        fallback_regimes=tuple(fallback_regimes),
        eps=eps,
    )


def calibrated_sensor_scores(
    per_channel_error: np.ndarray,
    feature_cols: Sequence[str],
    regime_ids: np.ndarray,
    stats: HealthySensorErrorStats,
) -> np.ndarray:
    """Z-score each sensor's error against ITS OWN healthy-region baseline.

    calibrated_j = (E_j - mean_healthy_j(regime)) / (std_healthy_j(regime) + eps)

    Unlike `per_sensor_contributions`/raw-unit error, this is NOT a
    fraction and can be negative (sensor reconstructing BETTER than
    its healthy baseline) -- see module docstring for why this is the
    one view immune to both the near-constant-sensor bias (normalized
    units) and the large-scale-sensor bias (raw units).

    Args:
        per_channel_error: Shape (n_windows, n_features). Either
            normalized- or raw-unit `per_channel_error` works and gives
            IDENTICAL results, as long as it is the same unit system
            `stats` was fit on (see module docstring).
        feature_cols: Column names for the `n_features` axis, in order.
        regime_ids: The regime assigned to each window, shape
            (n_windows,).
        stats: Output of `fit_healthy_sensor_error_stats`, fit on
            healthy calibration windows (never on the same windows
            being scored here).

    Returns:
        Array of shape (n_windows, len(stats.sensor_cols)): calibrated
        z-score per sensor, per window.

    Raises:
        ValueError: If shapes mismatch, `feature_cols` doesn't cover
            `stats.sensor_cols`, or a window's regime has no entry in
            `stats`.
    """
    per_channel_error = np.asarray(per_channel_error, dtype=float)
    feature_cols = list(feature_cols)
    regime_ids = np.asarray(regime_ids)
    sensor_cols = list(stats.sensor_cols)

    if per_channel_error.ndim != 2:
        raise ValueError(f"per_channel_error must be 2-D, got shape {per_channel_error.shape}")
    if per_channel_error.shape[1] != len(feature_cols):
        raise ValueError(
            f"per_channel_error has {per_channel_error.shape[1]} feature columns, "
            f"but feature_cols has {len(feature_cols)}"
        )
    if per_channel_error.shape[0] != len(regime_ids):
        raise ValueError(
            "per_channel_error and regime_ids must have the same number of windows, "
            f"got {per_channel_error.shape[0]} vs {len(regime_ids)}"
        )
    missing = set(sensor_cols) - set(feature_cols)
    if missing:
        raise ValueError(f"stats.sensor_cols not present in feature_cols: {sorted(missing)}")

    sensor_idx = [feature_cols.index(col) for col in sensor_cols]
    sensor_error = per_channel_error[:, sensor_idx]

    calibrated = np.empty_like(sensor_error)
    for regime in np.unique(regime_ids):
        regime_int = int(regime)
        if regime_int not in stats.mean:
            raise ValueError(f"No healthy sensor-error stats for regime {regime_int}")
        mask = regime_ids == regime
        calibrated[mask] = (sensor_error[mask] - stats.mean[regime_int]) / (stats.std[regime_int] + stats.eps)

    return calibrated


def _build_attribution(
    ranking_values_row: np.ndarray,
    raw_values_row: np.ndarray,
    sensor_cols: Sequence[str],
    engine_id: object,
    end_cycle: int,
    score: float,
    threshold: float,
    top_k: int,
) -> WindowAttribution:
    """Shared top-k ranking/packaging logic for a single window.

    Ranks `sensor_cols` by `ranking_values_row` descending (most-
    contributing/most-anomalous first), taking `raw_values_row` along
    for the ride as the `raw_errors` field. Used by both
    `attribute_window` (ranking = contribution fraction) and
    `attribute_window_calibrated` (ranking = calibrated z-score).
    """
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

    sensor_cols = list(sensor_cols)
    k = min(top_k, len(sensor_cols))
    order = np.argsort(ranking_values_row)[::-1][:k]

    return WindowAttribution(
        engine_id=engine_id,
        end_cycle=int(end_cycle),
        score=float(score),
        threshold=float(threshold),
        alert=bool(score > threshold),
        top_sensors=tuple(sensor_cols[i] for i in order),
        contributions=tuple(float(ranking_values_row[i]) for i in order),
        raw_errors=tuple(float(raw_values_row[i]) for i in order),
    )


def attribute_window(
    per_channel_error_row: np.ndarray,
    feature_cols: Sequence[str],
    engine_id: object,
    end_cycle: int,
    score: float,
    threshold: float,
    sensor_cols: Sequence[str] = SENSOR_COLUMNS,
    top_k: int = 5,
    eps: float = 1e-12,
) -> WindowAttribution:
    """Build a `WindowAttribution` for a single window.

    Args:
        per_channel_error_row: 1-D per-channel error for ONE window,
            shape (n_features,).
        feature_cols, sensor_cols, eps: See `per_sensor_contributions`.
        engine_id, end_cycle, score, threshold: Identify/describe this
            window (AI_CONTEXT.md G4 alert-object fields).
        top_k: Number of top-contributing sensors to retain. Must be a
            positive integer; clamped to `len(sensor_cols)` if larger.

    Returns:
        A `WindowAttribution` with `alert = score > threshold`.

    Raises:
        ValueError: If `top_k` is not a positive integer, or via
            `per_sensor_contributions` for shape/column mismatches.
    """
    contributions = per_sensor_contributions(
        per_channel_error_row.reshape(1, -1), feature_cols, sensor_cols, eps=eps
    )[0]
    sensor_cols = list(sensor_cols)
    feature_cols = list(feature_cols)
    sensor_idx = [feature_cols.index(col) for col in sensor_cols]
    raw_errors = np.asarray(per_channel_error_row, dtype=float)[sensor_idx]

    return _build_attribution(
        contributions, raw_errors, sensor_cols, engine_id, end_cycle, score, threshold, top_k
    )


def attribute_window_calibrated(
    per_channel_error_row: np.ndarray,
    feature_cols: Sequence[str],
    regime_id: int,
    stats: HealthySensorErrorStats,
    engine_id: object,
    end_cycle: int,
    score: float,
    threshold: float,
    top_k: int = 5,
) -> WindowAttribution:
    """Build a calibrated `WindowAttribution` for a single window.

    Ranks sensors by `calibrated_sensor_scores` (z-score of that
    sensor's error against its OWN healthy baseline) instead of raw
    contribution share -- see module docstring for why this is the
    scale-bias-free view. `contributions` here holds the calibrated
    z-score (can be negative), NOT a [0, 1] fraction.

    Args:
        per_channel_error_row: 1-D per-channel error for ONE window
            (either unit system -- see `calibrated_sensor_scores`).
        feature_cols: Column names for `per_channel_error_row`, in
            order.
        regime_id: This window's assigned regime (must have an entry
            in `stats`).
        stats: Output of `fit_healthy_sensor_error_stats`.
        engine_id, end_cycle, score, threshold, top_k: See
            `attribute_window`.

    Returns:
        A `WindowAttribution` whose `top_sensors` are ranked by
        calibrated z-score descending (most-anomalous-for-itself
        first), and whose `raw_errors` holds the underlying
        `per_channel_error_row` values (same unit system passed in).

    Raises:
        ValueError: If `top_k` is not positive, or via
            `calibrated_sensor_scores` for shape/regime mismatches.
    """
    calibrated = calibrated_sensor_scores(
        per_channel_error_row.reshape(1, -1), feature_cols, np.array([regime_id]), stats
    )[0]
    feature_cols = list(feature_cols)
    sensor_idx = [feature_cols.index(col) for col in stats.sensor_cols]
    raw_errors = np.asarray(per_channel_error_row, dtype=float)[sensor_idx]

    return _build_attribution(
        calibrated, raw_errors, list(stats.sensor_cols), engine_id, end_cycle, score, threshold, top_k
    )


def attribute_alerts(
    per_channel_error: np.ndarray,
    feature_cols: Sequence[str],
    engine_ids: np.ndarray,
    end_cycles: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray | float,
    alerts: np.ndarray,
    sensor_cols: Sequence[str] = SENSOR_COLUMNS,
    top_k: int = 5,
    eps: float = 1e-12,
) -> list[WindowAttribution]:
    """Attribute only the ALERTED windows (G4 explains alerts, not every window).

    Args:
        per_channel_error: Shape (n_windows, n_features), aligned with
            `engine_ids`/`end_cycles`/`scores`/`alerts`.
        feature_cols, sensor_cols, top_k, eps: See `attribute_window`.
        engine_ids, end_cycles, scores: Per-window metadata/score,
            shape (n_windows,) each.
        thresholds: Per-window threshold, shape (n_windows,), or a
            single scalar applied to every window (e.g. a fixed
            threshold rather than a regime-aware one).
        alerts: Boolean alert decision per window, shape (n_windows,).
            Only windows where this is True are attributed.

    Returns:
        A list of `WindowAttribution`, one per alerted window, in the
        same order the alerted windows appear in the input arrays.

    Raises:
        ValueError: If array lengths mismatch.
    """
    per_channel_error = np.asarray(per_channel_error, dtype=float)
    engine_ids = np.asarray(engine_ids)
    end_cycles = np.asarray(end_cycles)
    scores = np.asarray(scores, dtype=float)
    alerts = np.asarray(alerts).astype(bool)

    n = per_channel_error.shape[0]
    thresholds_arr = np.full(n, float(thresholds)) if np.ndim(thresholds) == 0 else np.asarray(thresholds, dtype=float)

    lengths = {n, len(engine_ids), len(end_cycles), len(scores), len(thresholds_arr), len(alerts)}
    if len(lengths) != 1:
        raise ValueError(
            "per_channel_error/engine_ids/end_cycles/scores/thresholds/alerts must "
            f"all have the same length, got lengths {lengths}"
        )

    alert_positions = np.flatnonzero(alerts)
    return [
        attribute_window(
            per_channel_error[i],
            feature_cols,
            engine_id=engine_ids[i],
            end_cycle=end_cycles[i],
            score=scores[i],
            threshold=thresholds_arr[i],
            sensor_cols=sensor_cols,
            top_k=top_k,
            eps=eps,
        )
        for i in alert_positions
    ]


def attribute_alerts_calibrated(
    per_channel_error: np.ndarray,
    feature_cols: Sequence[str],
    regime_ids: np.ndarray,
    stats: HealthySensorErrorStats,
    engine_ids: np.ndarray,
    end_cycles: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray | float,
    alerts: np.ndarray,
    top_k: int = 5,
) -> list[WindowAttribution]:
    """Calibrated counterpart to `attribute_alerts` (see module docstring).

    Args:
        per_channel_error: Shape (n_windows, n_features). Any unit
            system `stats` was fit on (normalized or raw -- identical
            result either way, see module docstring).
        feature_cols: Column names for `per_channel_error`, in order.
        regime_ids: The regime assigned to each window, shape
            (n_windows,).
        stats: Output of `fit_healthy_sensor_error_stats`, fit on
            healthy calibration windows only.
        engine_ids, end_cycles, scores, thresholds, alerts, top_k: See
            `attribute_alerts`.

    Returns:
        A list of `WindowAttribution`, one per alerted window, ranked
        by calibrated z-score (`contributions` field) rather than
        contribution fraction.

    Raises:
        ValueError: If array lengths mismatch.
    """
    per_channel_error = np.asarray(per_channel_error, dtype=float)
    regime_ids = np.asarray(regime_ids)
    engine_ids = np.asarray(engine_ids)
    end_cycles = np.asarray(end_cycles)
    scores = np.asarray(scores, dtype=float)
    alerts = np.asarray(alerts).astype(bool)

    n = per_channel_error.shape[0]
    thresholds_arr = np.full(n, float(thresholds)) if np.ndim(thresholds) == 0 else np.asarray(thresholds, dtype=float)

    lengths = {n, len(regime_ids), len(engine_ids), len(end_cycles), len(scores), len(thresholds_arr), len(alerts)}
    if len(lengths) != 1:
        raise ValueError(
            "per_channel_error/regime_ids/engine_ids/end_cycles/scores/thresholds/alerts must "
            f"all have the same length, got lengths {lengths}"
        )

    alert_positions = np.flatnonzero(alerts)
    return [
        attribute_window_calibrated(
            per_channel_error[i],
            feature_cols,
            regime_id=int(regime_ids[i]),
            stats=stats,
            engine_id=engine_ids[i],
            end_cycle=end_cycles[i],
            score=scores[i],
            threshold=thresholds_arr[i],
            top_k=top_k,
        )
        for i in alert_positions
    ]


def aggregate_sensor_frequency(
    attributions: Sequence[WindowAttribution],
    rank: int | None = 0,
) -> dict[str, int]:
    """Count how often each sensor appears among the top contributors.

    Used to summarize an experiment's alerts into "which sensors most
    often drive an anomaly alert" (AI_CONTEXT.md G4 example output:
    `sensor_7 0.31`, `sensor_11 0.27`, ... ranked across the alert
    population, not just one window).

    Args:
        attributions: Output of `attribute_alerts`.
        rank: If an int, counts only each attribution's sensor at that
            rank position (0 = the single top contributor). If None,
            counts a sensor once per attribution it appears ANYWHERE in
            `top_sensors` (i.e. within that attribution's `top_k`).

    Returns:
        Dict mapping sensor name -> count, sorted descending by count.
        Empty dict if `attributions` is empty.
    """
    counts: dict[str, int] = {}
    for attribution in attributions:
        if not attribution.top_sensors:
            continue
        sensors = (attribution.top_sensors[rank],) if rank is not None else attribution.top_sensors
        for sensor in sensors:
            counts[sensor] = counts.get(sensor, 0) + 1

    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


__all__ = [
    "WindowAttribution",
    "per_sensor_contributions",
    "rescale_to_raw_units",
    "HealthySensorErrorStats",
    "fit_healthy_sensor_error_stats",
    "calibrated_sensor_scores",
    "attribute_window",
    "attribute_alerts",
    "attribute_window_calibrated",
    "attribute_alerts_calibrated",
    "aggregate_sensor_frequency",
]
