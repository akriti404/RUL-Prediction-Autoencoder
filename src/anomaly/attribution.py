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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.schema import SENSOR_COLUMNS


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
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

    contributions = per_sensor_contributions(
        per_channel_error_row.reshape(1, -1), feature_cols, sensor_cols, eps=eps
    )[0]
    sensor_cols = list(sensor_cols)
    feature_cols = list(feature_cols)
    sensor_idx = [feature_cols.index(col) for col in sensor_cols]
    raw_errors = np.asarray(per_channel_error_row, dtype=float)[sensor_idx]

    k = min(top_k, len(sensor_cols))
    order = np.argsort(contributions)[::-1][:k]

    return WindowAttribution(
        engine_id=engine_id,
        end_cycle=int(end_cycle),
        score=float(score),
        threshold=float(threshold),
        alert=bool(score > threshold),
        top_sensors=tuple(sensor_cols[i] for i in order),
        contributions=tuple(float(contributions[i]) for i in order),
        raw_errors=tuple(float(raw_errors[i]) for i in order),
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
    "attribute_window",
    "attribute_alerts",
    "aggregate_sensor_frequency",
]
