"""Conformal calibration: nonconformity scores, quantile, dynamic/regime-aware threshold.

See AI_CONTEXT.md Section 11.3 (conformal calibration target) and G3
(statistically calibrated confidence/false-alarm guarantees).

Design decisions locked in for this project (confirmed with Aryaman,
Sep 2026):

  - Calibration set: the existing healthy `val_split` is REUSED as the
    calibration set (no dedicated third split). This is a standard,
    defensible split-conformal setup as long as val was never used to
    fit the model or any other threshold.
  - Nonconformity score: `normalized_anomaly_scores` (the
    variance-normalized, sign-corrected score already used for fixed-
    threshold detection), NOT raw `window_score`, to stay consistent
    with the rest of the pipeline.
  - Regime-conditioned: a SEPARATE threshold is calibrated per
    operating regime, rather than one pooled threshold, per the
    project's regime-conditioning novelty claim (G2/G3 combined).
  - Per-engine aggregation: within each regime, calibration uses ONE
    score per engine (that engine's mean normalized score among its
    windows assigned to that regime), not one score per window.
    Windows from the same engine's trajectory are highly correlated
    (consecutive, overlapping), so treating every window as an
    independent exchangeable calibration point would overstate the
    effective sample size and invalidate the coverage guarantee.
    Aggregating to one score per engine treats ENGINES as the
    exchangeable unit instead, which is the theoretically defensible
    choice here.

Coverage guarantee caveat (AI_CONTEXT.md Section G3 / Failure 7):
state this as "designed to provide the specified coverage under its
stated assumptions" (exchangeability between calibration and future
healthy data), never as an unconditional guarantee.

BUG FIX: `fit_conformal_thresholds_per_regime` originally only fit a
threshold for regimes that actually appeared in the calibration (val)
set. With a small val split (e.g. FD001), a regime can be entirely
absent from calibration — not just under `min_engines`, but zero
windows — and later appear at test time with no threshold at all,
raising in `apply_conformal_thresholds` instead of falling back. Fixed
by accepting an explicit `all_regimes` (the regime model's full
`range(n_regimes)`), so every possible regime gets either its own
threshold or the pooled fallback, never nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConformalThresholds:
    """Fitted per-regime conformal thresholds.

    Attributes:
        target_coverage: The coverage level calibration was fit for
            (e.g. 0.95).
        thresholds: Per-regime threshold value. A window in regime `r`
            is alerted if its score exceeds `thresholds[r]`.
        n_calibration_engines: Number of calibration engines
            contributing to each regime's threshold (diagnostic —
            small values mean that regime's threshold is less
            statistically reliable).
        fallback_regimes: Regimes whose threshold fell back to the
            pooled (all-regime) threshold because fewer than
            `min_engines` calibration engines were available for that
            regime specifically.
        pooled_threshold: The pooled (regime-agnostic) threshold,
            computed as a fallback and for reference/diagnostics.
        min_engines: The `min_engines` cutoff used when fitting.
    """

    target_coverage: float
    thresholds: dict[int, float]
    n_calibration_engines: dict[int, int]
    fallback_regimes: tuple[int, ...]
    pooled_threshold: float
    min_engines: int = field(default=2)


def _conformal_quantile(scores: Sequence[float], target_coverage: float) -> float:
    """Split-conformal quantile: the smallest value guaranteeing the
    target coverage among `n` exchangeable calibration scores.

    Uses the standard finite-sample-corrected order statistic:
    k = ceil((n + 1) * target_coverage); threshold = sorted(scores)[k-1].

    If `k > n` (too few calibration points to achieve the target
    coverage at all), returns +inf — no finite threshold can honestly
    claim that coverage with this little data, so no window is ever
    alertable at that threshold rather than silently under-covering.

    Args:
        scores: Calibration nonconformity scores (already aggregated
            to one score per exchangeable unit — here, per engine).
        target_coverage: Desired coverage in (0, 1), e.g. 0.95.

    Returns:
        The conformal threshold value (possibly +inf).
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    sorted_scores = np.sort(np.asarray(scores, dtype=float))
    k = ceil((n + 1) * target_coverage)
    if k > n:
        return float("inf")
    return float(sorted_scores[k - 1])


def fit_conformal_thresholds_per_regime(
    calibration_scores: np.ndarray,
    engine_ids: np.ndarray,
    regime_ids: np.ndarray,
    target_coverage: float = 0.95,
    min_engines: int = 2,
    all_regimes: Sequence[int] | None = None,
) -> ConformalThresholds:
    """Fit a separate conformal threshold per operating regime.

    For each regime, the calibration set is the set of ENGINES that
    have at least one window assigned to that regime; each such
    engine contributes a single nonconformity score (its mean
    `calibration_scores` among its windows in that regime — see
    module docstring for why engines, not windows, are the
    exchangeable unit here).

    If a regime has fewer than `min_engines` calibration engines, its
    threshold falls back to the POOLED threshold (computed the same
    way but across all engines regardless of regime), since a
    per-regime quantile from e.g. 1 engine is not a meaningful
    calibration and would silently produce an unreliable (often
    infinite or trivially low) threshold.

    Args:
        calibration_scores: 1-D array of window-level
            `normalized_anomaly_scores`, computed on healthy
            calibration (val) windows only.
        engine_ids: Engine ID per window, same length as
            `calibration_scores`.
        regime_ids: Operating-regime assignment per window (e.g. from
            `regime_model.model.predict` on each window's settings),
            same length as `calibration_scores`.
        target_coverage: Desired coverage in (0, 1). Defaults to 0.95
            per `configs/conformal_regime_ae.yaml`.
        min_engines: Minimum number of distinct engines a regime must
            have in the calibration set before it gets its own
            threshold; below this, falls back to the pooled threshold.
            Defaults to 2.
        all_regimes: The FULL set of regime IDs the regime model can
            ever produce (e.g. `range(regime_model.n_regimes)`), not
            just the regimes that happen to appear in this calibration
            set. Pass this explicitly whenever the calibration set may
            not cover every regime the model can output (e.g. a small
            val split) — any regime in `all_regimes` with ZERO
            calibration windows gets the pooled fallback threshold
            just like a too-small regime does, instead of being left
            out entirely and crashing later in `apply_conformal_thresholds`
            when it shows up at test/inference time. If None (default),
            only regimes observed in `regime_ids` get a threshold —
            safe only if you can guarantee every possible regime will
            appear in the calibration set.

    Returns:
        A `ConformalThresholds` instance covering every regime in
        `all_regimes` (or, if not given, every regime present in
        `regime_ids`).

    Raises:
        ValueError: If array lengths mismatch, `target_coverage` is
            not in (0, 1), or `min_engines` is not a positive integer.
    """
    calibration_scores = np.asarray(calibration_scores, dtype=float)
    engine_ids = np.asarray(engine_ids)
    regime_ids = np.asarray(regime_ids)

    if not (len(calibration_scores) == len(engine_ids) == len(regime_ids)):
        raise ValueError(
            "calibration_scores, engine_ids, and regime_ids must have the same "
            f"length, got {len(calibration_scores)}, {len(engine_ids)}, {len(regime_ids)}"
        )
    if not (0.0 < target_coverage < 1.0):
        raise ValueError(f"target_coverage must be in (0, 1), got {target_coverage!r}")
    if not isinstance(min_engines, int) or min_engines < 1:
        raise ValueError(f"min_engines must be a positive integer, got {min_engines!r}")

    # Pooled per-engine scores (regime-agnostic), used both as the
    # fallback and for diagnostics.
    pooled_per_engine = np.array(
        [calibration_scores[engine_ids == eid].mean() for eid in np.unique(engine_ids)]
    )
    pooled_threshold = _conformal_quantile(pooled_per_engine, target_coverage)

    thresholds: dict[int, float] = {}
    n_calibration_engines: dict[int, int] = {}
    fallback_regimes: list[int] = []

    regimes_to_fit = (
        sorted(set(int(r) for r in all_regimes) | set(int(r) for r in np.unique(regime_ids)))
        if all_regimes is not None
        else [int(r) for r in np.unique(regime_ids)]
    )

    for regime in regimes_to_fit:
        regime_mask = regime_ids == regime
        regime_engines = np.unique(engine_ids[regime_mask])
        n_engines = len(regime_engines)
        n_calibration_engines[int(regime)] = n_engines

        if n_engines < min_engines:
            thresholds[int(regime)] = pooled_threshold
            fallback_regimes.append(int(regime))
            continue

        per_engine_scores = np.array(
            [
                calibration_scores[regime_mask & (engine_ids == eid)].mean()
                for eid in regime_engines
            ]
        )
        thresholds[int(regime)] = _conformal_quantile(per_engine_scores, target_coverage)

    return ConformalThresholds(
        target_coverage=target_coverage,
        thresholds=thresholds,
        n_calibration_engines=n_calibration_engines,
        fallback_regimes=tuple(fallback_regimes),
        pooled_threshold=pooled_threshold,
        min_engines=min_engines,
    )


def apply_conformal_thresholds(
    scores: np.ndarray,
    regime_ids: np.ndarray,
    conformal: ConformalThresholds,
) -> np.ndarray:
    """Apply fitted per-regime conformal thresholds to produce alerts.

    Args:
        scores: 1-D array of window-level `normalized_anomaly_scores`
            to evaluate (e.g. test windows).
        regime_ids: Operating-regime assignment per window, same
            length as `scores`.
        conformal: A `ConformalThresholds` from
            `fit_conformal_thresholds_per_regime`.

    Returns:
        Boolean alert array, same shape as `scores`: True where the
        score exceeds that window's regime-specific threshold.

    Raises:
        ValueError: If `scores`/`regime_ids` shapes mismatch, or a
            regime in `regime_ids` has no fitted threshold (should not
            happen if regimes come from the same fitted regime model
            used for calibration, since KMeans.predict only ever
            returns known cluster IDs).
    """
    scores = np.asarray(scores, dtype=float)
    regime_ids = np.asarray(regime_ids)

    if scores.shape != regime_ids.shape:
        raise ValueError(
            f"scores and regime_ids must have the same shape, got {scores.shape} and {regime_ids.shape}"
        )

    missing = set(int(r) for r in np.unique(regime_ids)) - set(conformal.thresholds.keys())
    if missing:
        raise ValueError(
            f"regime_ids contain regime(s) with no fitted conformal threshold: {sorted(missing)}"
        )

    threshold_per_window = np.array([conformal.thresholds[int(r)] for r in regime_ids])
    return scores > threshold_per_window


def compute_calibration_diagnostics(
    calibration_scores: np.ndarray,
    engine_ids: np.ndarray,
    regime_ids: np.ndarray,
    conformal: ConformalThresholds,
) -> dict:
    """Empirical coverage diagnostics on the calibration set itself.

    Reports, per regime, the WINDOW-level false-alarm rate on the
    calibration data at the fitted threshold — i.e. what fraction of
    healthy calibration windows would themselves be (incorrectly)
    alerted. This should be in the neighborhood of `1 - target_coverage`
    at the ENGINE level (by construction); the window-level number is
    reported too since it's what actually feeds operational false-alarm
    rate, and will generally differ from the engine-level figure.

    This is a sanity-check report, not a substitute for the proper
    coverage guarantee statement in AI_CONTEXT.md Section G3 — see
    `Failure 7` in that document for the phrasing to use/avoid.

    Returns:
        A dict with keys: `target_coverage`, `pooled_threshold`,
        `min_engines`, and `per_regime`, the latter a dict keyed by
        regime ID with `threshold`, `n_calibration_engines`,
        `used_fallback`, `n_calibration_windows`, and
        `window_level_alert_rate`.
    """
    alerts = apply_conformal_thresholds(calibration_scores, regime_ids, conformal)

    per_regime = {}
    for regime in np.unique(regime_ids):
        mask = regime_ids == regime
        per_regime[int(regime)] = {
            "threshold": conformal.thresholds[int(regime)],
            "n_calibration_engines": conformal.n_calibration_engines.get(int(regime)),
            "used_fallback": int(regime) in conformal.fallback_regimes,
            "n_calibration_windows": int(mask.sum()),
            "window_level_alert_rate": float(alerts[mask].mean()) if mask.sum() > 0 else None,
        }

    return {
        "target_coverage": conformal.target_coverage,
        "pooled_threshold": conformal.pooled_threshold,
        "min_engines": conformal.min_engines,
        "per_regime": per_regime,
    }


__all__ = [
    "ConformalThresholds",
    "fit_conformal_thresholds_per_regime",
    "apply_conformal_thresholds",
    "compute_calibration_diagnostics",
]
