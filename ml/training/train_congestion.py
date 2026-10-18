"""Train the congestion classifier: gradient boosting vs a small MLP, register the better one.

    docker compose run --rm ml-trainer python -m ml.training.train_congestion \
        [--real-hours 24 | --no-real] [--synthetic-days 3] [--seed 0] [--no-promote]

Labels: synthetic ground truth plus weak labels on real data (ml.labels). Split by time
within each camera (70/15/15). Both models are tracked as child runs of one comparison
run in the MLflow experiment "congestion-classifier"; the one with the higher test
macro-F1 is registered as "congestion-classifier" and gets the "champion" alias.
"""

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.build import build_dataset
from ml.features import CONGESTION_FEATURES, compute_references
from ml.labels import CONGESTION_LEVELS
from ml.models.congestion import (
    CongestionPyfunc,
    MLPClassifier,
    feature_matrix,
    macro_f1,
    permutation_importance,
    train_gbm,
)
from ml.training.common import ML_DIR, log, pip_requirements, register, setup, time_split

EXPERIMENT = "congestion-classifier"
MODEL_NAME = "congestion-classifier"


def evaluate(model, X: np.ndarray, y: np.ndarray, origin: np.ndarray, out: Path, tag: str) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    pred = model.predict(X)
    labels = list(range(len(CONGESTION_LEVELS)))
    report = classification_report(
        y, pred, labels=labels, target_names=CONGESTION_LEVELS, output_dict=True, zero_division=0
    )
    metrics = {"test_accuracy": report["accuracy"], "test_macro_f1": macro_f1(y, pred)}
    for level in CONGESTION_LEVELS:
        for k in ("precision", "recall", "f1-score"):
            metrics[f"test_{level}_{k.replace('-score', '')}"] = report[level][k]
    for o in np.unique(origin):
        mask = origin == o
        metrics[f"test_macro_f1_{o}"] = macro_f1(y[mask], pred[mask])

    cm = confusion_matrix(y, pred, labels=labels)
    pd.DataFrame(cm, index=CONGESTION_LEVELS, columns=CONGESTION_LEVELS).to_csv(
        out / "confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CONGESTION_LEVELS).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"{tag}: test confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    (out / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return metrics


def importance(model, X, y, out: Path, tag: str, seed: int) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(4000, len(X)), replace=False)
    imp = permutation_importance(model, X[idx], y[idx], CONGESTION_FEATURES, repeats=3, seed=seed)
    imp.to_csv(out / "feature_importance.csv", index=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#4C78A8")
    ax.set_xlabel("drop in test macro-F1 when shuffled")
    ax.set_title(f"{tag}: permutation importance (top 15)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=120)
    plt.close(fig)
    return imp


def log_candidate(
    tag: str,
    kind: str,
    model,
    params: dict,
    refs,
    data,
    version,
    seed: int,
    extra_metrics: dict | None = None,
) -> tuple[str, dict]:
    import joblib
    import mlflow

    X_test, y_test, origin_test = data
    with mlflow.start_run(run_name=tag, nested=True) as run, tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        mlflow.log_params({"model": kind, **params, **version.mlflow_params()})
        metrics = evaluate(model, X_test, y_test, origin_test, out, tag)
        metrics.update(extra_metrics or {})
        mlflow.log_metrics(metrics)
        imp = importance(model, X_test, y_test, out, tag, seed)
        mlflow.log_artifacts(str(out), artifact_path="evaluation")
        log.info(
            "%s: test macro-F1 %.3f (real %.3f, synthetic %.3f); top features %s",
            tag,
            metrics["test_macro_f1"],
            metrics.get("test_macro_f1_real", float("nan")),
            metrics.get("test_macro_f1_synthetic", float("nan")),
            ", ".join(imp["feature"].head(4)),
        )

        model_path = out / ("model.joblib" if kind == "gbm" else "model.pt")
        if kind == "gbm":
            joblib.dump(model, model_path)
        else:
            model.save(model_path)
        meta = {
            "kind": kind,
            "features": CONGESTION_FEATURES,
            "levels": CONGESTION_LEVELS,
            "references": {k: v.to_dict() for k, v in refs.items()},
        }
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CongestionPyfunc(),
            artifacts={"model": str(model_path), "meta": str(out / "meta.json")},
            code_paths=[str(ML_DIR)],
            pip_requirements=pip_requirements(),
        )
        return run.info.run_id, metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--real-hours", type=float, default=None, help="default: all real data")
    parser.add_argument("--no-real", action="store_true", help="train on synthetic data only")
    parser.add_argument("--synthetic-days", type=float, default=3.0)
    parser.add_argument("--synthetic-cameras", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlp-epochs", type=int, default=40)
    parser.add_argument("--no-promote", action="store_true", help="don't register or set @champion")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    """Train both candidates and (unless --no-promote) register the winner. Returns the
    winner, its run and metrics, and the dataset with its split column."""
    setup()
    import mlflow

    df, version, _ = build_dataset(
        "congestion",
        include_real=not args.no_real,
        real_hours=args.real_hours,
        synthetic_days=args.synthetic_days,
        synthetic_cameras=args.synthetic_cameras,
        seed=args.seed,
    )
    df["split"] = time_split(df, (0.7, 0.15, 0.15))
    refs = compute_references(df[df["split"] == "train"])
    feats = feature_matrix(df, refs)  # keeps the label and split columns
    level_index = {level: i for i, level in enumerate(CONGESTION_LEVELS)}
    parts = {}
    for name in ("train", "val", "test"):
        part = feats[feats["split"] == name]
        parts[name] = (
            part[CONGESTION_FEATURES].to_numpy(np.float32),
            part["congestion"].map(level_index).to_numpy(),
            part["origin"].to_numpy(),
        )
    (X_tr, y_tr, _), (X_va, y_va, _), test = parts["train"], parts["val"], parts["test"]
    log.info(
        "rows train/val/test: %d / %d / %d; train classes %s",
        len(X_tr),
        len(X_va),
        len(test[0]),
        dict(zip(CONGESTION_LEVELS, np.bincount(y_tr, minlength=4).tolist(), strict=True)),
    )

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"compare-{datetime.now(UTC):%Y%m%d-%H%M%S}"):
        version.log_to_mlflow()
        mlflow.log_params(
            {
                "n_features": len(CONGESTION_FEATURES),
                "seed": args.seed,
                "rows_train": len(X_tr),
                "rows_val": len(X_va),
                "rows_test": len(test[0]),
            }
        )

        gbm = train_gbm(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]), seed=args.seed)
        gbm_run, gbm_metrics = log_candidate(
            "gradient-boosting",
            "gbm",
            gbm,
            {
                "max_iter": gbm.max_iter,
                "n_iter": gbm.n_iter_,
                "learning_rate": gbm.learning_rate,
                "max_leaf_nodes": gbm.max_leaf_nodes,
                "class_weight": "balanced",
            },
            refs,
            test,
            version,
            args.seed,
        )

        mlp = MLPClassifier(len(CONGESTION_FEATURES))
        history = mlp.fit(X_tr, y_tr, X_va, y_va, epochs=args.mlp_epochs, seed=args.seed)
        mlp_run, mlp_metrics = log_candidate(
            "mlp",
            "mlp",
            mlp,
            {
                "hidden": str(mlp.hidden),
                "dropout": mlp.dropout,
                "epochs_run": len(history),
                "max_epochs": args.mlp_epochs,
                "optimizer": "adam",
                "lr": 1e-3,
                "batch": 512,
                "class_weights": "inverse_frequency",
            },
            refs,
            test,
            version,
            args.seed,
            {"best_val_macro_f1": max(h["val_macro_f1"] for h in history)},
        )

        candidates = {"gradient-boosting": (gbm_run, gbm_metrics), "mlp": (mlp_run, mlp_metrics)}
        winner = max(candidates, key=lambda k: candidates[k][1]["test_macro_f1"])
        comparison = {
            k: {
                m: round(v[1][m], 4)
                for m in (
                    "test_macro_f1",
                    "test_accuracy",
                    "test_macro_f1_real",
                    "test_macro_f1_synthetic",
                )
                if m in v[1]
            }
            for k, v in candidates.items()
        }
        mlflow.log_dict(comparison, "comparison.json")
        mlflow.set_tags({"winner": winner, "winner_run_id": candidates[winner][0]})
        mlflow.log_metric("winner_test_macro_f1", candidates[winner][1]["test_macro_f1"])
        log.info("comparison: %s -> winner %s", json.dumps(comparison), winner)

        if not args.no_promote:
            v = register(
                candidates[winner][0],
                MODEL_NAME,
                "champion",
                f"{winner} congestion classifier, test macro-F1 "
                f"{candidates[winner][1]['test_macro_f1']:.3f}",
                {"model": winner, "dataset_version": version.version},
            )
            mlflow.set_tag("registered_version", v)
    registered = None
    if not args.no_promote:
        from ml.monitoring.reference import save_for_run

        registered = v
        save_for_run(MODEL_NAME, v, candidates[winner][0])
    return {
        "winner": winner,
        "run_id": candidates[winner][0],
        "metrics": candidates[winner][1],
        "comparison": comparison,
        "dataset": df,
        "dataset_version": version.version,
        "registered_version": registered,
    }


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "winner": result["winner"],
                "comparison": result["comparison"],
                "dataset": result["dataset_version"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
