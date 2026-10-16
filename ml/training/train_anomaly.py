"""Train the LSTM autoencoder anomaly detector and register it.

    docker compose run --rm ml-trainer python -m ml.training.train_anomaly \
        [--real-hours 24 | --no-real] [--synthetic-days 3] [--epochs 10] \
        [--threshold-percentile 99.5] [--seed 0] [--no-promote]

Training uses normal periods only: sequences containing a labelled anomaly or any part of
a scenario run are excluded. Split by time within each camera (60/20/20); the threshold
is a percentile of scores on normal validation sequences. Evaluated on synthetic test
episodes, on false alarms in real data, and on scenario_runs ground truth. Tracked in the
MLflow experiment "traffic-anomaly-detector" and registered under that name, alias
"champion".
"""

import argparse
import json
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.alarm import AlarmTracker
from ml.data.build import build_dataset
from ml.evaluation.anomaly_eval import episode_metrics, scenario_report
from ml.features import (
    AE_FEATURES,
    GLOBAL,
    SEQUENCE_LENGTH,
    WINDOW_S,
    base_features,
    build_sequences,
    compute_references,
    sequence_segments,
)
from ml.models.anomaly import (
    AnomalyConfig,
    AnomalyPyfunc,
    Normalizer,
    calibrate,
    reconstruct,
    save_artifacts,
    score_frame,
    score_parts,
    sequence_errors,
    train_autoencoder,
    window_levels,
)
from ml.training.common import ML_DIR, log, pip_requirements, register, setup, time_split

EXPERIMENT = "traffic-anomaly-detector"
MODEL_NAME = "traffic-anomaly-detector"
TRACKER = {
    "cooldown_s": AlarmTracker.cooldown_s,
    "clear_ratio": AlarmTracker.clear_ratio,
    "clear_after": AlarmTracker.clear_after,
    "min_consecutive": AlarmTracker.min_consecutive,
}


def normal_sequences(
    base: pd.DataFrame, z: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequences whose rows are all normal (is_anomaly == 0), the camera of each, and the
    row index each one ends at."""
    seqs, ends = build_sequences(z, sequence_segments(base), SEQUENCE_LENGTH, stride)
    bad = np.r_[0, np.cumsum((base["is_anomaly"].to_numpy() != 0).astype(int))]
    clean = (bad[ends + 1] - bad[ends + 1 - SEQUENCE_LENGTH]) == 0
    return seqs[clean], base["camera_id"].to_numpy()[ends][clean].astype(str), ends[clean]


def scored_table(sf) -> pd.DataFrame:
    """Rows with ``score`` = raw score / the camera's threshold, so 1.0 is the alarm level
    for every camera and the evaluation can replay alarms with threshold 1.0."""
    out = sf.base[["camera_id", "ts", "origin", "is_anomaly", "episode_id", "anomaly_type"]].copy()
    out["raw_score"] = sf.scores
    out["score"] = sf.scores / sf.thresholds
    return out


def plot_scores(val_ratio: np.ndarray, test: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    hi = max(np.nanmax(test["score"]) if len(test) else 1.0, 2.0)
    bins = np.logspace(np.log10(max(1e-3, np.nanmin(val_ratio))), np.log10(hi), 60)
    ax.hist(val_ratio, bins=bins, alpha=0.6, label="validation (normal)", density=True)
    anomalous = test.loc[test["is_anomaly"] == 1, "score"].dropna()
    if len(anomalous):
        ax.hist(anomalous, bins=bins, alpha=0.6, label="test anomalies", density=True)
    ax.axvline(1.0, color="red", linestyle="--", label="camera threshold")
    ax.set_xscale("log")
    ax.set_xlabel("score / camera threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--real-hours", type=float, default=None, help="default: all real data")
    parser.add_argument("--no-real", action="store_true", help="train on synthetic data only")
    parser.add_argument("--synthetic-days", type=float, default=3.0)
    parser.add_argument("--synthetic-cameras", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2, help="training sequence stride")
    parser.add_argument("--threshold-percentile", type=float, default=99.5)
    parser.add_argument("--score-method", choices=["max_scaled", "mean"], default="max_scaled")
    parser.add_argument(
        "--level-weight",
        type=float,
        default=1.0,
        help="weight of the distance-from-baseline term (0 disables it)",
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-promote", action="store_true", help="don't register or set @champion")
    args = parser.parse_args()
    setup()
    import mlflow

    df, version, windows = build_dataset(
        "anomaly",
        include_real=not args.no_real,
        real_hours=args.real_hours,
        synthetic_days=args.synthetic_days,
        synthetic_cameras=args.synthetic_cameras,
        seed=args.seed + 1000,  # a different synthetic draw from the congestion dataset
    )
    df["split"] = time_split(df, (0.6, 0.2, 0.2))
    cfg = AnomalyConfig(
        hidden=args.hidden,
        latent=args.latent,
        score_method=args.score_method,
        threshold_percentile=args.threshold_percentile,
        level_weight=args.level_weight,
    )

    train_raw = df[df["split"] == "train"]
    refs = compute_references(train_raw[train_raw["is_anomaly"] == 0])
    base_train = base_features(train_raw, refs)
    normalizer = Normalizer.fit(base_train[base_train["is_anomaly"] == 0])
    train_seqs, _, _ = normal_sequences(base_train, normalizer.transform(base_train), args.stride)
    base_val = base_features(df[df["split"] == "val"], refs)
    val_seqs, val_cams, val_ends = normal_sequences(base_val, normalizer.transform(base_val), 1)
    val_levels = window_levels(base_val, val_ends, cfg.score_steps)
    log.info(
        "normal sequences train/val: %d / %d (%d features x %d windows)",
        len(train_seqs),
        len(val_seqs),
        len(AE_FEATURES),
        SEQUENCE_LENGTH,
    )

    mlflow.set_experiment(EXPERIMENT)
    with (
        mlflow.start_run(run_name=f"lstm-ae-{datetime.now(UTC):%Y%m%d-%H%M%S}") as run,
        tempfile.TemporaryDirectory() as tmp,
    ):
        out = Path(tmp)
        version.log_to_mlflow()
        mlflow.log_params(
            {
                **asdict(cfg),
                "epochs": args.epochs,
                "stride": args.stride,
                "features": ",".join(AE_FEATURES),
                "train_sequences": len(train_seqs),
                "val_sequences": len(val_seqs),
                "seed": args.seed,
                **{f"alarm_{k}": v for k, v in TRACKER.items()},
            }
        )

        model, history = train_autoencoder(
            train_seqs,
            val_seqs,
            cfg,
            epochs=args.epochs,
            seed=args.seed,
            log=lambda h: (
                log.info("epoch %(epoch)d train %(train_loss).4f val %(val_loss).4f", h),
                mlflow.log_metrics(
                    {"train_loss": h["train_loss"], "val_loss": h["val_loss"]}, step=h["epoch"]
                ),
            ),
        )
        val_errors = sequence_errors(val_seqs, reconstruct(model, val_seqs), cfg.score_steps)
        calibration = calibrate(val_errors, val_cams, cfg, val_levels)
        val_parts = score_parts(val_errors, val_levels, val_cams, calibration, cfg)
        val_scores = val_parts.score
        mlflow.log_metric(
            "val_level_part_share",
            float((val_parts.level_score > val_parts.reconstruction_score).mean()),
        )
        val_ratio = val_scores / np.array([calibration.threshold(c) for c in val_cams])
        threshold = calibration.thresholds[GLOBAL]
        mlflow.log_metrics(
            {
                "threshold_global": threshold,
                "cameras_with_own_threshold": len(calibration.thresholds) - 1,
            }
        )
        mlflow.log_dict(calibration.thresholds, "evaluation/thresholds.json")
        log.info(
            "thresholds (p%.1f of normal validation scores, %s): %s",
            cfg.threshold_percentile,
            cfg.score_method,
            {k: round(v, 3) for k, v in calibration.thresholds.items()},
        )

        # --- synthetic test episodes
        test_raw = df[df["split"] == "test"]
        metrics = {}
        syn = test_raw[test_raw["origin"] == "synthetic"]
        if len(syn):
            scored = scored_table(score_frame(syn, refs, normalizer, model, cfg, calibration))
            m, by_type, per_episode = episode_metrics(scored, scored, 1.0, **TRACKER)
            metrics.update({f"synthetic_{k}": v for k, v in m.items()})
            for row in by_type.itertuples():
                metrics[f"synthetic_recall_{row.type}"] = row.recall
                if pd.notna(row.delay_median_s):
                    metrics[f"synthetic_delay_median_s_{row.type}"] = row.delay_median_s
            by_type.to_csv(out / "synthetic_by_type.csv", index=False)
            per_episode.to_csv(out / "synthetic_episodes.csv", index=False)
            plot_scores(val_ratio, scored, out / "score_distribution.png")
            log.info("synthetic test: %s", json.dumps({k: round(v, 3) for k, v in m.items()}))
            log.info("by type:\n%s", by_type.to_string(index=False))

        # --- real data: false alarms and scenario runs
        real = df[df["origin"] == "real"]
        if len(real):
            real_scored = scored_table(score_frame(real, refs, normalizer, model, cfg, calibration))
            test_cutoff = real.loc[real["split"] == "test", "ts"].min()
            real_test = real_scored[
                (real_scored["ts"] >= test_cutoff) & (real_scored["is_anomaly"] == 0)
            ]
            m, _, _ = episode_metrics(real_test, real_test, 1.0, **TRACKER)
            metrics["real_test_false_alarms"] = m["false_alarms"]
            metrics["real_test_false_alarms_per_hour"] = m["false_alarms_per_hour"]
            metrics["real_test_hours"] = len(real_test) * WINDOW_S / 3600
            if len(windows):
                report = scenario_report(real_scored, windows, 1.0, **TRACKER)
                report.to_csv(out / "scenario_runs.csv", index=False)
                evaluated = report[report["scored_windows"] > 0]
                if len(evaluated):
                    metrics["scenario_runs_evaluated"] = len(evaluated)
                    metrics["scenario_recall"] = float(evaluated["fired"].mean())
                    if evaluated["fired"].any():
                        metrics["scenario_delay_mean_s"] = float(evaluated["delay_s"].mean())
                        metrics["scenario_delay_scenario_time_mean_s"] = float(
                            evaluated["delay_scenario_s"].mean()
                        )
                log.info(
                    "scenario runs:\n%s",
                    report[
                        [
                            "scenario",
                            "camera_id",
                            "time_scale",
                            "scored_windows",
                            "fired",
                            "alarms",
                            "delay_s",
                            "delay_scenario_s",
                        ]
                    ].to_string(index=False),
                )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
        (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
        mlflow.log_artifacts(str(out), artifact_path="evaluation")

        artifacts = save_artifacts(out / "model", model, normalizer, refs, cfg, calibration)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=AnomalyPyfunc(),
            artifacts=artifacts,
            code_paths=[str(ML_DIR)],
            pip_requirements=pip_requirements(),
        )
        # Fail before registering if the packaged model doesn't load or predict.
        check = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
        sample = df[df["camera_id"] == df["camera_id"].iloc[-1]].tail(SEQUENCE_LENGTH + 4)
        result = check.predict(sample)
        log.info(
            "pyfunc check: %s",
            result[["camera_id", "valid", "score", "threshold"]].to_dict("records"),
        )

        if not args.no_promote:
            v = register(
                run.info.run_id,
                MODEL_NAME,
                "champion",
                f"LSTM autoencoder ({cfg.score_method}), per-camera thresholds "
                f"(p{cfg.threshold_percentile}), global {threshold:.4f}",
                {"dataset_version": version.version},
            )
            mlflow.set_tag("registered_version", v)
    print(
        json.dumps(
            {
                "threshold": threshold,
                "dataset": version.version,
                **{k: round(float(v), 4) for k, v in metrics.items() if pd.notna(v)},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
