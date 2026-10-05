"""Experiment entry point: conformal-calibrated regime-conditioned model.

See AI_CONTEXT.md Section 11.3 (conformal calibration target), G3
(statistically calibrated false-alarm guarantees), and `src/anomaly/
conformal.py` for the fitting/application logic and the design
decisions it documents (calibration set = reused val split, score =
`normalized_anomaly_scores`, regime-conditioned, per-engine
aggregation).

This script does NOT retrain a model. It loads an existing checkpoint
produced by `run_regime_conditioned.py` (which saves
`model_state_dict`, `regime_model`, `normalization_stats`,
`feature_cols`, and architecture dims), rebuilds the model, and:

  1. Rebuilds the SAME healthy val split used during that model's
     training (same fd_id/val_frac/healthy_frac/seed — pass these
     explicitly if the checkpoint wasn't trained with the defaults
     below, or results will not reflect the actual calibration set
     the model saw at val time).
  2. Scores val windows -> normalized_anomaly_scores -> fits per-regime
     conformal thresholds (src/anomaly/conformal.py).
  3. Optionally evaluates on test, using those thresholds.

Per-window regime assignment (both val and test) is computed directly
from each window's own LAST-cycle operating settings via
`regime_model.model.predict`, matching how `run_regime_conditioned.py`
feeds `train_settings`/`val_settings` to the model (window's last row),
rather than threading a separate `operating_regime` label column
through `create_windows`.

Run from repo root (after a regime-conditioned checkpoint exists, e.g.
`results/checkpoints/fd001_ae_regime_conditioned_v001.pt`):

    python -m src.experiments.run_conformal \\
        --checkpoint-path results/checkpoints/fd001_ae_regime_conditioned_v001.pt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.anomaly.conformal import (
    apply_conformal_thresholds,
    compute_calibration_diagnostics,
    fit_conformal_thresholds_per_regime,
)
from src.anomaly.reconstruction import normalized_anomaly_scores, window_scores_numpy
from src.data.healthy_region import select_healthy_region
from src.data.loaders import load_test, load_test_rul, load_train
from src.data.normalization import transform_by_regime
from src.data.regimes import RegimeModel
from src.data.schema import SENSOR_COLUMNS, SETTING_COLUMNS
from src.data.splits import split_by_engine
from src.data.windows import create_windows
from src.evaluation.evaluation_runner import (
    compute_engine_total_life,
    evaluate_detection,
    label_anomalous_by_life_fraction,
)
from src.models.conditioned_autoencoder import RegimeConditionedAutoencoder

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _window_regimes(window_X: np.ndarray, regime_model: RegimeModel) -> np.ndarray:
    """Assign each window to a regime using its own last-cycle settings.

    Matches the conditioning input `run_regime_conditioned.py` feeds
    the model (`window_X[:, -1, :3]`), so the regime a window is
    calibrated/evaluated under is the same regime the model itself
    conditioned on for that window.
    """
    last_settings = window_X[:, -1, :3]
    return regime_model.model.predict(last_settings)


def run_conformal_experiment(
    checkpoint_path: str,
    fd_id: str = "FD001",
    val_frac: float = 0.15,
    healthy_frac: float = 0.85,
    window_size: int = 30,
    stride: int = 1,
    hidden_dims: tuple[int, ...] = (64, 32),
    regime_hidden_dims: tuple[int, ...] = (16,),
    seed: int = 42,
    target_coverage: float = 0.85,
    min_engines: int = 2,
    persistence: int = 1,
    experiment_name: str = "fd001_ae_conformal_v001",
    evaluate_on_test: bool = True,
) -> dict:
    """Fit regime-conditioned conformal thresholds from a trained checkpoint
    and optionally evaluate on test.

    Args:
        checkpoint_path: Path to a `.pt` checkpoint saved by
            `run_regime_conditioned.py`.
        fd_id, val_frac, healthy_frac, window_size, stride, seed: Must
            match whatever the checkpoint's model was actually trained
            with, so the reconstructed val split is the SAME val split
            the model saw during training (not accidentally leaking
            different engines into calibration).
        hidden_dims, regime_hidden_dims: Architecture dims for
            rebuilding the model shell before loading weights; must
            match the checkpoint's trained architecture.
        target_coverage, min_engines: Passed to
            `fit_conformal_thresholds_per_regime`.

    Returns:
        A dict summary (also written to
        `results/logs/<experiment_name>.json`).
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
        "evaluation": {"persistence": persistence},
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

    # 2. Rebuild the SAME healthy val split the checkpoint was trained with.
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
        raise ValueError(
            f"window_size={window_size} produced zero calibration (val) windows."
        )

    val_X = torch.tensor(val_windows.X, dtype=torch.float32)
    val_settings = torch.tensor(val_windows.X[:, -1, :3], dtype=torch.float32)

    with torch.no_grad():
        val_recon, _ = model(val_X, val_settings)
    val_raw_scores = window_scores_numpy(val_X, val_recon)
    val_scores = normalized_anomaly_scores(val_windows.X, val_raw_scores)
    val_regimes = _window_regimes(val_windows.X, regime_model)

    # 3. Fit per-regime conformal thresholds on the calibration (val) set.
    conformal = fit_conformal_thresholds_per_regime(
        calibration_scores=val_scores,
        engine_ids=val_windows.engine_ids,
        regime_ids=val_regimes,
        target_coverage=target_coverage,
        min_engines=min_engines,
        all_regimes=range(regime_model.n_regimes),
    )
    calibration_diagnostics = compute_calibration_diagnostics(
        val_scores, val_windows.engine_ids, val_regimes, conformal
    )

    summary = {
        **config,
        "n_calibration_windows": len(val_windows),
        "n_calibration_engines_total": int(len(np.unique(val_windows.engine_ids))),
        "pooled_threshold": conformal.pooled_threshold,
        "per_regime_thresholds": conformal.thresholds,
        "fallback_regimes": list(conformal.fallback_regimes),
        "calibration_diagnostics": calibration_diagnostics,
    }

    # 4. Optional test-set evaluation (test RUL used only for labeling,
    #    never for threshold fitting — AI_CONTEXT.md Section 17 Rule 5).
    if evaluate_on_test:
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
            summary["test_evaluation_warning"] = (
                f"window_size={window_size} produced zero test windows."
            )
        else:
            test_X = torch.tensor(test_windows.X, dtype=torch.float32)
            test_settings = torch.tensor(test_windows.X[:, -1, :3], dtype=torch.float32)

            with torch.no_grad():
                test_recon, _ = model(test_X, test_settings)
            test_raw_scores = window_scores_numpy(test_X, test_recon)
            test_scores = normalized_anomaly_scores(test_windows.X, test_raw_scores)
            test_regimes = _window_regimes(test_windows.X, regime_model)

            test_alerts = apply_conformal_thresholds(test_scores, test_regimes, conformal)
            test_y_true = test_windows.y.flatten()

            total_life = compute_engine_total_life(test_df, test_rul)
            eval_result = evaluate_detection(
                y_true=test_y_true,
                scores=test_scores,
                alerts=test_alerts,
                engine_ids=test_windows.engine_ids,
                end_cycles=test_windows.end_cycles,
                engine_total_life=total_life,
                persistence=persistence,
            )

            summary["test_evaluation"] = {
                "n_test_windows": len(test_windows),
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "f1": eval_result.f1,
                "roc_auc": eval_result.roc_auc,
                "false_alarm_rate": eval_result.false_alarm_rate,
                "detection_rate": eval_result.detection_rate,
                "mean_lead_time": eval_result.mean_lead_time,
                "n_engines": eval_result.n_engines,
            }

    # 5. Persist experiment log (no new checkpoint — model wasn't retrained).
    logs_dir = _REPO_ROOT / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{experiment_name}.json"
    log_path.write_text(json.dumps(summary, indent=2))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--fd-id", default="FD001")
    parser.add_argument("--experiment-name", default="fd001_ae_conformal_v001")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--healthy-frac", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-coverage", type=float, default=0.95)
    parser.add_argument("--min-engines", type=int, default=2)
    parser.add_argument("--persistence", type=int, default=1)
    parser.add_argument("--no-test-eval", action="store_true", help="Skip test-set evaluation.")
    args = parser.parse_args()

    summary = run_conformal_experiment(
        checkpoint_path=args.checkpoint_path,
        fd_id=args.fd_id,
        val_frac=args.val_frac,
        healthy_frac=args.healthy_frac,
        window_size=args.window_size,
        stride=args.stride,
        seed=args.seed,
        target_coverage=args.target_coverage,
        min_engines=args.min_engines,
        persistence=args.persistence,
        experiment_name=args.experiment_name,
        evaluate_on_test=not args.no_test_eval,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
