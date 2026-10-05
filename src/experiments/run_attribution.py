"""Experiment entry point: per-sensor reconstruction-error attribution (G4).

See AI_CONTEXT.md Section 19 (`anomaly/attribution.py` module
contract), G4 (per-sensor attribution), and `src/experiments/
run_conformal.py` (this script reuses its checkpoint-loading /
val-split / conformal-fitting flow rather than retraining or
recalibrating independently -- the attributed alerts should be the
SAME alerts the conformal-threshold experiment reports, just with an
explanation attached to each one).

This script does NOT retrain a model and does NOT fit a new
threshold. It loads an existing regime-conditioned checkpoint (from
`run_regime_conditioned.py`), refits the per-regime conformal
thresholds on the same healthy val split used during training (per
`run_conformal.py`), scores the test set, and for every ALERTED test
window:

  1. Computes the per-channel (per-sensor) reconstruction error
     (`anomaly/reconstruction.per_channel_error`), in the NORMALIZED
     (per-regime z-score) units the model actually operates in.
  2. Rescales that error back to RAW (physical sensor-unit) error
     (`anomaly/attribution.rescale_to_raw_units`), since a channel
     with tiny natural within-regime variance gets its normalized
     error inflated by dividing by its own tiny std -- confirmed on
     FD002, where `sensor_6` (within-regime std ~0.0034) topped the
     normalized ranking despite not being an informative sensor in
     C-MAPSS literature.
  3. Calibrates each sensor's error against ITS OWN healthy-region
     baseline (`anomaly/attribution.fit_healthy_sensor_error_stats` /
     `attribute_alerts_calibrated`), fit on the same healthy val
     windows used for conformal calibration. This turned out to be
     necessary because RAW units traded the near-constant-sensor bias
     for the opposite one: on FD002, raw-unit attribution collapsed
     onto just the 4 highest-variance sensors (sensor_3/4/9/14) for
     nearly every alert, since raw error scales with a sensor's own
     physical variance regardless of reconstruction quality. The
     calibrated view is immune to both biases (see the module
     docstring in `anomaly/attribution.py` for the full derivation).
  4. Ranks sensors by their contribution to that window's total error,
     separately in each of the three views
     (`anomaly/attribution.attribute_window` / `attribute_window_calibrated`).
  5. Records engine ID, end cycle, score, threshold, and the top
     contributing sensors under all three views.

It then aggregates across all alerts to report which sensors most
often drive an alert (AI_CONTEXT.md G4 example output: "sensor_7 0.31,
sensor_11 0.27, ..."), separately for normalized, raw, and calibrated
units, and saves a bar chart of each. THE CALIBRATED VIEW is the one
to trust as the primary ranking; normalized/raw are kept as a visible
cross-check, since the discrepancy between them is itself informative
(a sensor that tops the calibrated ranking AND one of the other two is
a stronger signal than one that only tops the calibrated ranking).

Run from repo root (after a regime-conditioned checkpoint exists):

    python -m src.experiments.run_attribution \\
        --checkpoint-path results/checkpoints/fd002_ae_regime_v001.pt \\
        --fd-id FD002 \\
        --experiment-name fd002_ae_attribution_v001
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.anomaly.attribution import (
    aggregate_sensor_frequency,
    attribute_alerts,
    attribute_alerts_calibrated,
    fit_healthy_sensor_error_stats,
    rescale_to_raw_units,
)
from src.anomaly.conformal import apply_conformal_thresholds, fit_conformal_thresholds_per_regime
from src.anomaly.reconstruction import normalized_anomaly_scores, per_channel_error, window_scores_numpy
from src.data.healthy_region import select_healthy_region
from src.data.loaders import load_test, load_test_rul, load_train
from src.data.normalization import transform_by_regime
from src.data.regimes import RegimeModel
from src.data.schema import SENSOR_COLUMNS
from src.data.splits import split_by_engine
from src.data.windows import create_windows
from src.evaluation.evaluation_runner import label_anomalous_by_life_fraction
from src.models.conditioned_autoencoder import RegimeConditionedAutoencoder

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _window_regimes(window_X: np.ndarray, regime_model: RegimeModel) -> np.ndarray:
    """Assign each window to a regime using its own last-cycle settings.

    Matches `run_conformal.py`'s `_window_regimes` so the same
    windows get the same regime assignment (and therefore the same
    threshold / alert decision) as the conformal-threshold experiment.
    """
    last_settings = window_X[:, -1, :3]
    return regime_model.model.predict(last_settings)


def _plot_sensor_frequency(frequency: dict[str, int], out_path: Path, title: str) -> None:
    """Bar chart of alert counts per top-contributing sensor.

    AI_CONTEXT.md Section 23: anomaly-detection visualization should
    include "top sensor contributions." Saved as a static PNG --
    reusable across experiments without any plotting infrastructure
    beyond matplotlib (already a project dependency).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frequency:
        return

    sensors = list(frequency.keys())
    counts = list(frequency.values())

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(sensors))))
    ax.barh(sensors[::-1], counts[::-1], color="steelblue")
    ax.set_xlabel("Number of alerts where this sensor was the top contributor")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_attribution_experiment(
    checkpoint_path: str,
    fd_id: str = "FD001",
    val_frac: float = 0.15,
    healthy_frac: float = 0.85,
    window_size: int = 30,
    stride: int = 1,
    hidden_dims: tuple[int, ...] = (64, 32),
    regime_hidden_dims: tuple[int, ...] = (16,),
    seed: int = 42,
    target_coverage: float = 0.95,
    min_engines: int = 2,
    min_calibration_windows: int = 30,
    top_k: int = 5,
    n_examples: int = 10,
    experiment_name: str = "fd001_ae_attribution_v001",
) -> dict:
    """Fit conformal thresholds, alert on test windows, and explain each alert.

    Args:
        checkpoint_path: Path to a `.pt` checkpoint from
            `run_regime_conditioned.py`.
        fd_id, val_frac, healthy_frac, window_size, stride, seed: Must
            match the checkpoint's own training configuration (see
            `run_conformal.py` -- same constraint applies here since
            this script rebuilds the same val split).
        hidden_dims, regime_hidden_dims: Architecture dims for
            rebuilding the model shell before loading weights.
        target_coverage, min_engines: Passed to
            `fit_conformal_thresholds_per_regime`.
        min_calibration_windows: Passed to
            `fit_healthy_sensor_error_stats` as `min_windows` -- the
            minimum healthy (val) windows a regime needs before it gets
            its own per-sensor error baseline, else falls back to the
            pooled baseline.
        top_k: Number of top-contributing sensors to report per alert.
        n_examples: Number of individual alert explanations to persist
            in full in the output JSON (the aggregate sensor-frequency
            summary covers every alert regardless of this value).

    Returns:
        A dict summary (also written to
        `results/logs/<experiment_name>.json`), and a bar-chart PNG at
        `results/figures/<experiment_name>_sensor_frequency.png`.
    """
    config = {
        "experiment_id": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": fd_id,
        "checkpoint_path": checkpoint_path,
        "split": {"val_frac": val_frac, "seed": seed},
        "healthy_region": {"healthy_frac": healthy_frac},
        "window": {"size": window_size, "stride": stride},
        "threshold": {
            "method": "conformal",
            "target_coverage": target_coverage,
            "min_engines": min_engines,
            "calibration_set": "reused_val_split",
            "score": "normalized_anomaly_scores",
            "aggregation": "per_engine_mean_within_regime",
        },
        "attribution": {
            "top_k": top_k,
            "sensor_cols": SENSOR_COLUMNS,
            "min_calibration_windows": min_calibration_windows,
        },
    }

    # 1. Load checkpoint (no retraining).
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    feature_cols: list[str] = checkpoint["feature_cols"]
    regime_model: RegimeModel = checkpoint["regime_model"]
    normalization_stats = checkpoint["normalization_stats"]
    latent_dim = checkpoint["latent_dim"]
    regime_embedding_dim = checkpoint["regime_embedding_dim"]
    n_features = checkpoint.get("n_features", len(feature_cols))
    ckpt_window_size = checkpoint.get("window_size", window_size)

    if ckpt_window_size != window_size:
        raise ValueError(
            f"window_size={window_size} does not match checkpoint's trained "
            f"window_size={ckpt_window_size}; pass --window-size {ckpt_window_size}."
        )

    model = RegimeConditionedAutoencoder(
        window_size=window_size,
        n_features=n_features,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        regime_hidden_dims=regime_hidden_dims,
        regime_embedding_dim=regime_embedding_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Rebuild the SAME healthy val split used at training time, and
    #    fit conformal thresholds from it (mirrors run_conformal.py).
    train_full = load_train(fd_id)
    _, val_split = split_by_engine(train_full, val_frac=val_frac, seed=seed)
    val_healthy = select_healthy_region(val_split, healthy_frac=healthy_frac)

    val_healthy = val_healthy.copy()
    val_healthy[feature_cols] = val_healthy[feature_cols].astype(float)
    val_healthy["operating_regime"] = regime_model.model.predict(
        val_healthy[list(regime_model.feature_cols)].to_numpy(dtype=float)
    )
    val_normalized = transform_by_regime(val_healthy, normalization_stats)

    val_windows = create_windows(
        val_normalized, window_size=window_size, stride=stride, feature_cols=feature_cols
    )
    if len(val_windows) == 0:
        raise ValueError(f"window_size={window_size} produced zero calibration (val) windows.")

    val_X = torch.tensor(val_windows.X, dtype=torch.float32)
    val_settings = torch.tensor(val_windows.X[:, -1, :3], dtype=torch.float32)

    with torch.no_grad():
        val_recon, _ = model(val_X, val_settings)
    val_raw_scores = window_scores_numpy(val_X, val_recon)
    val_scores = normalized_anomaly_scores(val_windows.X, val_raw_scores)
    val_regimes = _window_regimes(val_windows.X, regime_model)

    # Per-sensor healthy baseline for calibrated attribution (see
    # module docstring / anomaly/attribution.py), fit on these SAME
    # healthy val windows -- never on test data.
    val_channel_error_normalized = per_channel_error(val_X, val_recon).detach().cpu().numpy()
    sensor_error_stats = fit_healthy_sensor_error_stats(
        val_channel_error_normalized,
        feature_cols,
        val_regimes,
        sensor_cols=SENSOR_COLUMNS,
        min_windows=min_calibration_windows,
        all_regimes=range(regime_model.n_regimes),
    )

    conformal = fit_conformal_thresholds_per_regime(
        calibration_scores=val_scores,
        engine_ids=val_windows.engine_ids,
        regime_ids=val_regimes,
        target_coverage=target_coverage,
        min_engines=min_engines,
        all_regimes=range(regime_model.n_regimes),
    )

    # 3. Score the test set and apply the fitted thresholds.
    test_df = load_test(fd_id)
    test_rul = load_test_rul(fd_id)

    test_labeled = label_anomalous_by_life_fraction(test_df, test_rul, healthy_frac=healthy_frac)
    test_labeled = test_labeled.copy()
    test_labeled[feature_cols] = test_labeled[feature_cols].astype(float)
    test_labeled["operating_regime"] = regime_model.model.predict(
        test_labeled[list(regime_model.feature_cols)].to_numpy(dtype=float)
    )
    test_normalized = transform_by_regime(test_labeled, normalization_stats)

    test_windows = create_windows(
        test_normalized,
        window_size=window_size,
        stride=stride,
        feature_cols=feature_cols,
        label_cols=["is_anomalous"],
    )
    if len(test_windows) == 0:
        raise ValueError(f"window_size={window_size} produced zero test windows.")

    test_X = torch.tensor(test_windows.X, dtype=torch.float32)
    test_settings = torch.tensor(test_windows.X[:, -1, :3], dtype=torch.float32)

    with torch.no_grad():
        test_recon, _ = model(test_X, test_settings)
    test_raw_scores = window_scores_numpy(test_X, test_recon)
    test_scores = normalized_anomaly_scores(test_windows.X, test_raw_scores)
    test_regimes = _window_regimes(test_windows.X, regime_model)
    test_alerts = apply_conformal_thresholds(test_scores, test_regimes, conformal)
    test_thresholds = np.array([conformal.thresholds[int(r)] for r in test_regimes])

    # Per-channel (per-sensor) error, in NORMALIZED units -- the channel
    # axis preserved by `anomaly/reconstruction.py` specifically for
    # attribution (AI_CONTEXT.md Section 10).
    test_channel_error_normalized = per_channel_error(test_X, test_recon).detach().cpu().numpy()

    # Rescaled to RAW (physical sensor-unit) error -- see module
    # docstring and `anomaly/attribution.rescale_to_raw_units` for why
    # the normalized ranking alone can be misleading for near-constant
    # sensors.
    test_channel_error_raw = rescale_to_raw_units(
        test_channel_error_normalized, feature_cols, test_regimes, normalization_stats
    )

    true_positive_mask = test_windows.y.flatten().astype(bool)[np.flatnonzero(test_alerts)]

    def _attribute(channel_error: np.ndarray) -> tuple[list, dict, dict, dict]:
        attributions = attribute_alerts(
            per_channel_error=channel_error,
            feature_cols=feature_cols,
            engine_ids=test_windows.engine_ids,
            end_cycles=test_windows.end_cycles,
            scores=test_scores,
            thresholds=test_thresholds,
            alerts=test_alerts,
            sensor_cols=SENSOR_COLUMNS,
            top_k=top_k,
        )
        top1 = aggregate_sensor_frequency(attributions, rank=0)
        any_rank = aggregate_sensor_frequency(attributions, rank=None)
        true_positive_attributions = [a for a, is_tp in zip(attributions, true_positive_mask) if is_tp]
        true_positive_top1 = aggregate_sensor_frequency(true_positive_attributions, rank=0)
        return attributions, top1, any_rank, true_positive_top1

    # 4. Attribute every alerted window, in all THREE views.
    attributions_normalized, top1_normalized, any_rank_normalized, tp_top1_normalized = _attribute(
        test_channel_error_normalized
    )
    attributions_raw, top1_raw, any_rank_raw, tp_top1_raw = _attribute(test_channel_error_raw)

    attributions_calibrated = attribute_alerts_calibrated(
        per_channel_error=test_channel_error_normalized,
        feature_cols=feature_cols,
        regime_ids=test_regimes,
        stats=sensor_error_stats,
        engine_ids=test_windows.engine_ids,
        end_cycles=test_windows.end_cycles,
        scores=test_scores,
        thresholds=test_thresholds,
        alerts=test_alerts,
        top_k=top_k,
    )
    top1_calibrated = aggregate_sensor_frequency(attributions_calibrated, rank=0)
    any_rank_calibrated = aggregate_sensor_frequency(attributions_calibrated, rank=None)
    tp_calibrated_attributions = [
        a for a, is_tp in zip(attributions_calibrated, true_positive_mask) if is_tp
    ]
    tp_top1_calibrated = aggregate_sensor_frequency(tp_calibrated_attributions, rank=0)

    examples = [
        {
            "engine_id": int(a_norm.engine_id),
            "end_cycle": a_norm.end_cycle,
            "score": a_norm.score,
            "threshold": a_norm.threshold,
            "top_sensors_normalized": list(a_norm.top_sensors),
            "contributions_normalized": [round(c, 4) for c in a_norm.contributions],
            "top_sensors_raw": list(a_raw.top_sensors),
            "contributions_raw": [round(c, 4) for c in a_raw.contributions],
            "top_sensors_calibrated": list(a_cal.top_sensors),
            "calibrated_zscores": [round(c, 4) for c in a_cal.contributions],
        }
        for a_norm, a_raw, a_cal in zip(
            attributions_normalized[:n_examples],
            attributions_raw[:n_examples],
            attributions_calibrated[:n_examples],
        )
    ]

    summary = {
        **config,
        "n_test_windows": len(test_windows),
        "n_alerts": int(test_alerts.sum()),
        "n_true_positive_alerts": int(true_positive_mask.sum()),
        "normalized_units": {
            "top1_sensor_frequency": top1_normalized,
            "any_rank_sensor_frequency": any_rank_normalized,
            "true_positive_top1_sensor_frequency": tp_top1_normalized,
        },
        "raw_units": {
            "top1_sensor_frequency": top1_raw,
            "any_rank_sensor_frequency": any_rank_raw,
            "true_positive_top1_sensor_frequency": tp_top1_raw,
        },
        "calibrated_units": {
            "note": "PRIMARY ranking -- immune to the scale bias affecting normalized_units/raw_units, see module docstrings",
            "fallback_regimes": list(sensor_error_stats.fallback_regimes),
            "top1_sensor_frequency": top1_calibrated,
            "any_rank_sensor_frequency": any_rank_calibrated,
            "true_positive_top1_sensor_frequency": tp_top1_calibrated,
        },
        "example_alerts": examples,
    }

    # 5. Persist experiment log + sensor-frequency charts (one per view).
    logs_dir = _REPO_ROOT / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{experiment_name}.json"
    log_path.write_text(json.dumps(summary, indent=2))

    figures_dir = _REPO_ROOT / "results" / "figures"
    _plot_sensor_frequency(
        top1_normalized,
        figures_dir / f"{experiment_name}_sensor_frequency_normalized.png",
        title=f"{experiment_name}: top sensor (NORMALIZED units) across {summary['n_alerts']} alerts",
    )
    _plot_sensor_frequency(
        top1_raw,
        figures_dir / f"{experiment_name}_sensor_frequency_raw.png",
        title=f"{experiment_name}: top sensor (RAW units) across {summary['n_alerts']} alerts",
    )
    _plot_sensor_frequency(
        top1_calibrated,
        figures_dir / f"{experiment_name}_sensor_frequency_calibrated.png",
        title=f"{experiment_name}: top sensor (CALIBRATED, primary) across {summary['n_alerts']} alerts",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--fd-id", default="FD001")
    parser.add_argument("--experiment-name", default="fd001_ae_attribution_v001")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--healthy-frac", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-coverage", type=float, default=0.95)
    parser.add_argument("--min-engines", type=int, default=2)
    parser.add_argument("--min-calibration-windows", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-examples", type=int, default=10)
    args = parser.parse_args()

    summary = run_attribution_experiment(
        checkpoint_path=args.checkpoint_path,
        fd_id=args.fd_id,
        val_frac=args.val_frac,
        healthy_frac=args.healthy_frac,
        window_size=args.window_size,
        stride=args.stride,
        seed=args.seed,
        target_coverage=args.target_coverage,
        min_engines=args.min_engines,
        min_calibration_windows=args.min_calibration_windows,
        top_k=args.top_k,
        n_examples=args.n_examples,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
