"""Train the incident text classifier (incident type + severity) and register the winner.

    docker compose run --rm ml-trainer python -m ml.training.train_nlp [--epochs 3]

Two candidates, each a nested MLflow run in experiment "incident-text-classifier":
- tfidf-baseline: TF-IDF (word 1-2 grams + char 2-5 grams) + logistic regression per task
- distilbert: distilbert-base-uncased, one shared encoder with an incident-type head and a
  severity head, fine-tuned on CPU (short max length, few epochs; the time is logged)

A stratified part of reports_train.csv is held out for validation (best-epoch selection);
reports_test.csv (unseen phrasing and location forms) is the final comparison. The model
with the higher mean test macro-F1 over both tasks is registered as
"incident-text-classifier" @champion (an MLflow pyfunc with tokenizer and label maps).

The parent run also evaluates the parts of the pipeline that aren't trained here: location
extraction (pretrained NER + gazetteer) against location_mention, and camera matching.
"""

import argparse
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.evaluation import nlp_eval
from ml.nlp.baseline import TfidfBaseline
from ml.nlp.labels import TASKS, label_maps
from ml.nlp.matching import CameraMatcher
from ml.nlp.pyfunc import IncidentTextClassifier, save_baseline_dir
from ml.training.common import ML_DIR, log, pip_requirements, register, setup

EXPERIMENT = "incident-text-classifier"
MODEL_NAME = "incident-text-classifier"
DATA = Path("data/nlp")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing = {"text", "incident_type", "severity", "camera_id", "location_mention", "source"}
    missing -= set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    return df.reset_index(drop=True)


def evaluate_candidate(name, predict, frames, y, maps, out: Path) -> dict[str, float]:
    """Metrics on val and test, plus test artifacts (reports, confusion, worst errors)."""
    import mlflow

    metrics = {}
    for split in ("val", "test"):
        started = time.perf_counter()
        probs = predict(frames[split].text.tolist())
        ms = (time.perf_counter() - started) / max(len(frames[split]), 1) * 1000
        m, reports = nlp_eval.classification_metrics(probs, y[split], maps, split)
        metrics.update(m, **{f"{split}_inference_ms_per_text": ms})
        if split == "test":
            nlp_eval.write_json(out / "classification_report_test.json", reports)
            nlp_eval.confusion_artifacts(probs, y[split], maps, out, f"{name} (test)")
            nlp_eval.worst_misclassified(frames[split], probs, y[split], maps).to_csv(
                out / "worst_misclassified_test.csv", index=False
            )
            nlp_eval.accuracy_by_source(frames[split], probs, y[split]).to_csv(
                out / "accuracy_by_source_test.csv", index=False
            )
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(str(out), artifact_path="evaluation")
    return metrics


def log_pyfunc(model_dir: Path, sample: list[str]) -> None:
    import mlflow

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=IncidentTextClassifier(),
        artifacts={"model_dir": str(model_dir)},
        code_paths=[str(ML_DIR)],
        pip_requirements=pip_requirements("requirements-runtime.txt", "requirements-nlp.txt"),
        input_example=pd.DataFrame({"text": sample[:2]}),
    )
    # Fail before registering if the packaged model doesn't load or predict.
    run_id = mlflow.active_run().info.run_id
    check = mlflow.pyfunc.load_model(f"runs:/{run_id}/model").predict(
        pd.DataFrame({"text": sample})
    )
    log.info("pyfunc check: %s", check[["incident_type", "severity"]].to_dict("records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", type=Path, default=DATA / "reports_train.csv")
    parser.add_argument("--test", type=Path, default=DATA / "reports_test.csv")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--ner-model", default=os.getenv("NER_MODEL", "dslim/bert-base-NER"))
    parser.add_argument("--threads", type=int, default=int(os.getenv("TRAIN_THREADS", "6")))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-transformer", action="store_true", help="baseline only")
    parser.add_argument("--skip-entities", action="store_true", help="no NER / matching eval")
    parser.add_argument("--no-promote", action="store_true", help="don't register @champion")
    parser.add_argument("--limit", type=int, default=None, help="subsample (smoke runs)")
    args = parser.parse_args()
    setup(args.threads)
    import mlflow

    maps = label_maps()
    train_full, test = load(args.train), load(args.test)
    if args.limit:
        train_full = train_full.sample(min(args.limit, len(train_full)), random_state=args.seed)
        test = test.head(max(20, args.limit // 4))
    train, val = train_test_split(
        train_full,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=train_full.incident_type + "/" + train_full.severity,
    )
    frames = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test,
    }
    y = {s: {t: maps[t].encode(f[t]) for t in TASKS} for s, f in frames.items()}
    log.info("train %d, val %d, test %d", *(len(f) for f in frames.values()))
    mlflow.set_experiment(EXPERIMENT)
    work = Path(tempfile.mkdtemp(prefix="nlp-"))
    results: dict[str, dict] = {}

    stamp = f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=f"nlp-{stamp}") as parent:
        mlflow.log_params(
            {
                "train_rows": len(frames["train"]),
                "val_rows": len(frames["val"]),
                "test_rows": len(test),
                "val_fraction": args.val_fraction,
                "threads": args.threads,
                "seed": args.seed,
                "selection": "higher mean test macro-F1 over both tasks",
            }
        )
        # ------------------------------------------------------------------ baseline
        with mlflow.start_run(run_name="tfidf-baseline", nested=True) as run:
            baseline = TfidfBaseline(seed=args.seed)
            mlflow.log_params({"model": "tfidf-logreg", **baseline.params()})
            started = time.perf_counter()
            baseline.fit(frames["train"].text.tolist(), y["train"])
            seconds = time.perf_counter() - started
            mlflow.log_metric("train_seconds", seconds)
            m = evaluate_candidate(
                "tfidf-baseline", baseline.predict_proba, frames, y, maps, work / "baseline-eval"
            )
            log_pyfunc(
                save_baseline_dir(baseline, maps, work / "baseline-model"),
                test.text.head(4).tolist(),
            )
            results["tfidf-baseline"] = {"run_id": run.info.run_id, "train_seconds": seconds, **m}
            log.info("baseline: test mean macro-F1 %.4f in %.1fs", m["test_mean_macro_f1"], seconds)

        # ------------------------------------------------------------------ DistilBERT
        if not args.skip_transformer:
            from ml.nlp import transformer

            cfg = transformer.TransformerConfig(
                base_model=args.base_model,
                max_length=args.max_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
            with mlflow.start_run(run_name="distilbert", nested=True) as run:
                mlflow.log_params(
                    {
                        "model": "distilbert-multitask",
                        **{f"bert_{k}": v for k, v in vars(cfg).items()},
                    }
                )
                started = time.perf_counter()
                model, tokenizer, history = transformer.train(
                    frames["train"].text.tolist(),
                    y["train"],
                    frames["val"].text.tolist(),
                    y["val"],
                    {t: len(maps[t]) for t in TASKS},
                    cfg,
                    on_epoch=lambda e, h: mlflow.log_metrics(
                        {k: v for k, v in h.items() if k != "epoch"}, step=e
                    ),
                )
                seconds = time.perf_counter() - started
                mlflow.log_metric("train_seconds", seconds)
                mlflow.log_dict({"history": history}, "evaluation/history.json")

                def predict(texts, model=model, tokenizer=tokenizer):
                    return transformer.predict_proba(model, tokenizer, texts, cfg.max_length)

                m = evaluate_candidate("distilbert", predict, frames, y, maps, work / "bert-eval")
                model_dir = transformer.save(model, tokenizer, cfg, maps, work / "bert-model")
                log_pyfunc(model_dir, test.text.head(4).tolist())
                results["distilbert"] = {"run_id": run.info.run_id, "train_seconds": seconds, **m}
                log.info(
                    "distilbert: test mean macro-F1 %.4f, trained in %.0fs",
                    m["test_mean_macro_f1"],
                    seconds,
                )

        # ------------------------------------------------------------------ comparison
        rows = []
        for name, r in results.items():
            for task in TASKS:
                rows.append(
                    {
                        "model": name,
                        "task": task,
                        "test_accuracy": r[f"test_{task}_accuracy"],
                        "test_macro_f1": r[f"test_{task}_macro_f1"],
                        "test_weighted_f1": r[f"test_{task}_weighted_f1"],
                        "val_macro_f1": r[f"val_{task}_macro_f1"],
                        "train_seconds": r["train_seconds"],
                        "inference_ms_per_text": r["test_inference_ms_per_text"],
                    }
                )
        table = pd.DataFrame(rows)
        winner = max(results, key=lambda n: results[n]["test_mean_macro_f1"])
        out = work / "comparison"
        out.mkdir()
        table.to_csv(out / "comparison.csv", index=False)
        mlflow.log_artifacts(str(out), artifact_path="comparison")
        mlflow.log_metrics(
            {f"{n}_test_mean_macro_f1": r["test_mean_macro_f1"] for n, r in results.items()}
        )
        mlflow.set_tag("winner", winner)

        # ------------------------------------------------------------------ entities, cameras
        entity_metrics: dict[str, float] = {}
        if not args.skip_entities:
            entity_metrics = evaluate_entities(test, args.ner_model, work / "entities")
            mlflow.log_metrics(entity_metrics)
            mlflow.log_artifacts(str(work / "entities"), artifact_path="entities")

        version = None
        if not args.no_promote:
            r = results[winner]
            version = register(
                r["run_id"],
                MODEL_NAME,
                "champion",
                f"{winner}: test macro-F1 incident_type {r['test_incident_type_macro_f1']:.3f}, "
                f"severity {r['test_severity_macro_f1']:.3f}",
                {"candidate": winner, "parent_run": parent.info.run_id},
            )
            mlflow.set_tag("registered_version", version)

    print("\n" + table.round(4).to_string(index=False))
    print(
        json.dumps(
            {
                "winner": winner,
                "registered_version": version,
                **{k: round(v, 4) for k, v in entity_metrics.items()},
            },
            indent=2,
        )
    )


def evaluate_entities(test: pd.DataFrame, ner_model: str, out: Path) -> dict[str, float]:
    from ml.nlp.ner import NerExtractor
    from traffic_common.db.seed import DEMO_CAMERAS
    from traffic_common.gazetteer import CameraInfo

    out.mkdir(parents=True, exist_ok=True)
    cameras = [
        CameraInfo(c["id"], c["name"], c["location_name"], c["latitude"], c["longitude"])
        for c in DEMO_CAMERAS
    ]
    matcher = CameraMatcher(cameras)
    extractor = NerExtractor(ner_model).load()  # not timed: download / load
    started = time.perf_counter()
    ner = extractor.extract(test.text.tolist())
    ner_ms = (time.perf_counter() - started) / len(test) * 1000
    gaz = [matcher.gazetteer_entities(t) for t in test.text]
    loc_metrics, loc_misses = nlp_eval.location_extraction_metrics(test, ner, gaz)
    cam_metrics, by_source, cam_misses = nlp_eval.camera_matching_metrics(test, matcher)
    loc_misses.to_csv(out / "location_misses.csv", index=False)
    by_source.to_csv(out / "camera_accuracy_by_source.csv", index=False)
    cam_misses.to_csv(out / "camera_misses.csv", index=False)
    labels = pd.Series([e["label"] for ents in ner for e in ents]).value_counts().to_dict()
    nlp_eval.write_json(out / "ner_label_counts.json", labels)
    return {**loc_metrics, **cam_metrics, "ner_ms_per_text": ner_ms}


if __name__ == "__main__":
    main()
