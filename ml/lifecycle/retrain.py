"""Retrain a model and promote the candidate only if it passes the evaluation gate.

    docker compose run --rm ml-trainer python -m ml.lifecycle.retrain <model> [options]

Models: congestion-classifier, traffic-anomaly-detector, incident-text-classifier.

1. **Train a candidate** with the model's usual training code, without promoting it.
2. **Evaluate the candidate and the champion on the same held-out data**, each through its
   own registered pyfunc (so each uses its own references, normalisation and thresholds).
3. **Decide** (ml.lifecycle.gate): promote only if the candidate beats the champion on the
   primary metric and regresses on no guard metric by more than its tolerance.
4. **Record** it: the candidate is registered as a new version with alias ``challenger`` (and
   ``champion`` if promoted; services reload within MODEL_REFRESH_SECONDS); the comparison and
   decision go to an MLflow run "promotion-gate-..." in the model's experiment and to the
   candidate's row in ``model_versions.promotion``; an open retraining recommendation is
   resolved with the decision. A promoted candidate also gets its drift reference.

Per model (held-out set; primary metric; guards):

- congestion-classifier: the new dataset's test split (time-based, last 15% per camera);
  macro-F1; accuracy, macro-F1 on real rows, recall of heavy and of standstill.
- traffic-anomaly-detector: synthetic test episodes and real test windows; event F1 on the
  synthetic episodes; recall, false alarms per hour (synthetic and real), median delay.
- incident-text-classifier: reports_test.csv and a held-out 30% of recent labelled production
  reports; mean macro-F1 of both tasks, averaged over the two sets; macro-F1 per task on
  reports_test.csv, mean macro-F1 on the recent reports.

The text candidate is trained on reports_train.csv plus the other 70% of recent labelled
production reports (simulator ground truth: the newest ``--recent-hours``, standard and
drift-mode styles alike), repeated ``--recent-weight`` times; by default DistilBERT is
fine-tuned from the champion for one epoch (``--from-scratch`` for the base model,
``--candidate tfidf`` for the fast baseline).
"""

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.lifecycle.gate import Decision, GatePolicy, MetricSpec, decide
from ml.lifecycle.records import record_promotion, resolve_recommendation

log = logging.getLogger("ml.lifecycle")
MODELS = ("congestion-classifier", "traffic-anomaly-detector", "incident-text-classifier")
NLP_DATA = Path("data/nlp")


def policy_for(model_name: str, tolerance: float, min_improvement: float) -> GatePolicy:
    if model_name == "congestion-classifier":
        primary = MetricSpec("macro_f1")
        guards = (
            MetricSpec("accuracy"),
            MetricSpec("macro_f1_real"),
            MetricSpec("recall_heavy", tolerance=max(tolerance, 0.02)),
            MetricSpec("recall_standstill", tolerance=max(tolerance, 0.02)),
        )
    elif model_name == "traffic-anomaly-detector":
        primary = MetricSpec("synthetic_event_f1")
        guards = (
            MetricSpec("synthetic_recall", tolerance=max(tolerance, 0.02)),
            MetricSpec("synthetic_false_alarms_per_hour", higher_is_better=False, tolerance=0.25),
            MetricSpec("real_false_alarms_per_hour", higher_is_better=False, tolerance=0.25),
            MetricSpec("synthetic_delay_median_s", higher_is_better=False, tolerance=10.0),
        )
    else:
        primary = MetricSpec("heldout_mean_macro_f1")
        guards = (
            MetricSpec("test_incident_type_macro_f1"),
            MetricSpec("test_severity_macro_f1"),
            MetricSpec("recent_mean_macro_f1"),
        )
    return GatePolicy(primary, guards, tolerance, min_improvement)


def _load(uri: str):
    import mlflow.pyfunc

    return mlflow.pyfunc.load_model(uri).unwrap_python_model()


# ----------------------------------------------------------------------------- congestion


def evaluate_congestion(python_model, df: pd.DataFrame) -> dict[str, float]:
    from sklearn.metrics import recall_score

    from ml.labels import CONGESTION_LEVELS
    from ml.models.congestion import feature_matrix, macro_f1

    feats = feature_matrix(df, python_model.refs)
    test = feats[feats["split"] == "test"]
    index = {level: i for i, level in enumerate(CONGESTION_LEVELS)}
    y = test["congestion"].map(index).to_numpy()
    pred = python_model.model.predict_proba(
        test[python_model.features].to_numpy(np.float32)
    ).argmax(axis=1)
    out = {
        "macro_f1": macro_f1(y, pred),
        "accuracy": float((y == pred).mean()),
        "rows": float(len(test)),
    }
    real = (test["origin"] == "real").to_numpy()
    if real.any():
        out["macro_f1_real"] = macro_f1(y[real], pred[real])
    recalls = recall_score(y, pred, labels=list(range(len(CONGESTION_LEVELS))), average=None,
                           zero_division=0)  # fmt: skip
    out.update(
        {f"recall_{lvl}": float(r) for lvl, r in zip(CONGESTION_LEVELS, recalls, strict=True)}
    )
    return out


def retrain_congestion(args, champion_uri: str | None) -> dict[str, Any]:
    from ml.training import train_congestion

    targs = train_congestion.parse_args(["--no-promote", "--seed", str(args.seed)])
    trained = train_congestion.run(targs)
    df = trained["dataset"]
    candidate = evaluate_congestion(_load(f"runs:/{trained['run_id']}/model"), df)
    champion = evaluate_congestion(_load(champion_uri), df) if champion_uri else None
    return {
        "run_id": trained["run_id"],
        "candidate": candidate,
        "champion": champion,
        "heldout": f"dataset {trained['dataset_version']} test split "
        f"({int(candidate['rows'])} rows)",
        "description": f"{trained['winner']} congestion classifier (retrained)",
        "tags": {"model": trained["winner"], "dataset_version": trained["dataset_version"]},
    }


# ----------------------------------------------------------------------------- anomaly


def evaluate_anomaly(python_model, df: pd.DataFrame) -> dict[str, float]:
    from ml.evaluation.anomaly_eval import episode_metrics
    from ml.models.anomaly import score_frame
    from ml.training.train_anomaly import TRACKER, scored_table

    def scored(rows: pd.DataFrame) -> pd.DataFrame:
        sf = score_frame(rows, python_model.refs, python_model.normalizer, python_model.model,
                         python_model.cfg, python_model.calibration)  # fmt: skip
        return scored_table(sf)

    out: dict[str, float] = {}
    test = df[df["split"] == "test"]
    syn = test[test["origin"] == "synthetic"]
    if len(syn):
        s = scored(syn)
        m, _, _ = episode_metrics(s, s, 1.0, **TRACKER)
        p, r = m["precision"], m["recall"]
        out.update(
            {
                "synthetic_recall": r,
                "synthetic_precision": p,
                "synthetic_event_f1": 2 * p * r / (p + r) if p and r and p + r > 0 else 0.0,
                "synthetic_false_alarms_per_hour": m["false_alarms_per_hour"],
                "synthetic_delay_median_s": m["delay_median_s"],
            }
        )
    real = df[df["origin"] == "real"]
    if len(real):
        s = scored(real)
        cutoff = real.loc[real["split"] == "test", "ts"].min()
        normal = s[(s["ts"] >= cutoff) & (s["is_anomaly"] == 0)]
        if len(normal):
            m, _, _ = episode_metrics(normal, normal, 1.0, **TRACKER)
            out["real_false_alarms_per_hour"] = m["false_alarms_per_hour"]
    return {k: float(v) for k, v in out.items() if v is not None and pd.notna(v)}


def retrain_anomaly(args, champion_uri: str | None) -> dict[str, Any]:
    from ml.training import train_anomaly

    targs = train_anomaly.parse_args(["--no-promote", "--seed", str(args.seed)])
    trained = train_anomaly.run(targs)
    df = trained["dataset"]
    candidate = evaluate_anomaly(_load(f"runs:/{trained['run_id']}/model"), df)
    champion = evaluate_anomaly(_load(champion_uri), df) if champion_uri else None
    return {
        "run_id": trained["run_id"],
        "candidate": candidate,
        "champion": champion,
        "heldout": f"dataset {trained['dataset_version']}: synthetic test episodes, "
        "real test windows",
        "description": f"LSTM autoencoder (retrained), global threshold {trained['threshold']:.4f}",
        "tags": {"dataset_version": trained["dataset_version"]},
    }


# ----------------------------------------------------------------------------- text

_RECENT_SQL = """
SELECT text, incident_type, severity, camera_id, COALESCE(location_mention, '') AS location_mention,
       source, text_style, emitted_at
FROM simulated_reports WHERE emitted_at >= :since ORDER BY emitted_at DESC LIMIT :limit
"""


def recent_reports(hours: float, limit: int) -> pd.DataFrame:
    """Labelled production reports (the simulator's ground truth for what it emitted)."""
    from sqlalchemy import text

    from traffic_common.config import get_settings
    from traffic_common.db import create_db_engine

    engine = create_db_engine(get_settings("ml-trainer").postgres_url)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(_RECENT_SQL),
            conn,
            params={"since": datetime.now(UTC) - timedelta(hours=hours), "limit": limit},
        )
    engine.dispose()
    return df.drop_duplicates("text").reset_index(drop=True)


def evaluate_text(classifier, sets: dict[str, pd.DataFrame]) -> dict[str, float]:
    from ml.evaluation.nlp_eval import classification_metrics
    from ml.nlp.labels import TASKS, label_maps

    maps = label_maps()
    out: dict[str, float] = {}
    for name, frame in sets.items():
        if frame is None or frame.empty:
            continue
        started = time.perf_counter()
        probs = classifier.predict_probabilities(frame["text"].astype(str).tolist())
        out[f"{name}_inference_ms_per_text"] = (time.perf_counter() - started) / len(frame) * 1e3
        y = {t: maps[t].encode(frame[t]) for t in TASKS}
        m, _ = classification_metrics(probs, y, maps, name)
        for t in TASKS:
            out[f"{name}_{t}_accuracy"] = m[f"{name}_{t}_accuracy"]
            out[f"{name}_{t}_macro_f1"] = m[f"{name}_{t}_macro_f1"]
        out[f"{name}_mean_macro_f1"] = m[f"{name}_mean_macro_f1"]
        out[f"{name}_rows"] = float(len(frame))
    means = [out[f"{n}_mean_macro_f1"] for n in sets if f"{n}_mean_macro_f1" in out]
    out["heldout_mean_macro_f1"] = float(np.mean(means)) if means else float("nan")
    return out


def retrain_text(args, champion_uri: str | None) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split

    from ml.training import train_nlp

    test = train_nlp.load(NLP_DATA / "reports_test.csv")
    recent = recent_reports(args.recent_hours, args.recent_max)
    recent_train = recent_holdout = pd.DataFrame()
    if len(recent) >= 20:
        recent_train, recent_holdout = train_test_split(
            recent, test_size=0.3, random_state=args.seed
        )
    styles = recent["text_style"].fillna("standard").value_counts().to_dict() if len(recent) else {}
    log.info(
        "recent labelled reports: %d (%s); %d for training, %d held out",
        len(recent),
        styles,
        len(recent_train),
        len(recent_holdout),
    )
    champion = _load(champion_uri) if champion_uri else None
    # Kept (not a temp file): the version's drift reference includes what it was trained on.
    work = Path("data/nlp/retrain")
    work.mkdir(parents=True, exist_ok=True)
    extra_path = work / f"recent_train-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.csv"
    targv = ["--no-promote", "--skip-entities", "--seed", str(args.seed)]
    if len(recent_train):
        columns = ["text", "incident_type", "severity", "camera_id", "location_mention", "source"]
        recent_train[columns].to_csv(extra_path, index=False)
        targv += ["--extra-train", str(extra_path),
                  "--extra-weight", str(args.recent_weight)]  # fmt: skip
    if args.candidate == "tfidf":
        targv.append("--skip-transformer")
    else:
        targv += ["--epochs", str(args.epochs)]
        warm = champion is not None and champion.kind == "transformer" and not args.from_scratch
        if warm:
            targv += ["--warm-start", champion_uri, "--learning-rate", str(args.learning_rate)]
    trained = train_nlp.run(train_nlp.parse_args(targv))
    name = "tfidf-baseline" if args.candidate == "tfidf" else "distilbert"
    run_id = trained["results"][name]["run_id"]
    sets = {"test": test, "recent": recent_holdout}
    candidate = evaluate_text(_load(f"runs:/{run_id}/model"), sets)
    champion_metrics = evaluate_text(champion, sets) if champion is not None else None
    heldout = f"reports_test.csv ({len(test)})"
    if len(recent_holdout):
        slang = int((recent_holdout["text_style"] == "slang").sum())
        heldout += f" + recent production holdout ({len(recent_holdout)}, {slang} drift-style)"
    return {
        "run_id": run_id,
        "candidate": candidate,
        "champion": champion_metrics,
        "heldout": heldout,
        "description": f"{name} (retrained with {len(recent_train)} recent labelled reports)",
        "tags": {
            "candidate": name,
            "parent_run": trained["parent_run_id"],
            "recent_reports": str(len(recent)),
        },  # fmt: skip
    }


# ----------------------------------------------------------------------------- gate


def run_gate(args) -> dict[str, Any]:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    from ml.training.common import setup
    from traffic_common.config import get_settings
    from traffic_common.db import create_db_engine

    setup(args.threads)
    client = MlflowClient()
    model = args.model
    try:
        champion_mv = client.get_model_version_by_alias(model, "champion")
        champion_version, champion_uri = (
            str(champion_mv.version),
            f"models:/{model}/{champion_mv.version}",
        )
    except MlflowException:
        champion_version, champion_uri = None, None
    log.info("%s: champion %s; training a candidate", model, champion_version or "none")
    started = time.perf_counter()
    adapter = {
        "congestion-classifier": retrain_congestion,
        "traffic-anomaly-detector": retrain_anomaly,
        "incident-text-classifier": retrain_text,
    }[model]
    out = adapter(args, champion_uri)
    policy = policy_for(model, args.tolerance, args.min_improvement)
    decision: Decision = decide(policy, out["champion"], out["candidate"])
    elapsed = time.perf_counter() - started

    # Register the candidate (alias "challenger"), and promote it if it passed.
    candidate_version = None
    if not args.dry_run:
        mv = mlflow.register_model(
            f"runs:/{out['run_id']}/model",
            model,
            tags={**out["tags"], "gate_decision": decision.decision, "reason": args.reason or ""},
        )
        candidate_version = str(mv.version)
        client.update_model_version(
            model,
            candidate_version,
            description=f"{out['description']}; gate: {decision.decision}",
        )
        client.set_registered_model_alias(model, "challenger", candidate_version)
        if decision.promoted:
            client.set_registered_model_alias(model, "champion", candidate_version)

    record = {
        **decision.to_dict(),
        "decided_at": datetime.now(UTC).isoformat(),
        "champion_version": champion_version,
        "candidate_version": candidate_version,
        "candidate_run_id": out["run_id"],
        "heldout": out["heldout"],
        "policy": {
            "primary": policy.primary.name,
            "guards": [g.name for g in policy.guards],
            "tolerance": policy.tolerance,
            "min_improvement": policy.min_improvement,
        },
        "reason_for_retraining": args.reason,
        "train_seconds": round(elapsed, 1),
        "dry_run": args.dry_run,
    }
    mlflow.set_experiment(model)
    with mlflow.start_run(run_name=f"promotion-gate-{datetime.now(UTC):%Y%m%d-%H%M%S}") as run:
        mlflow.set_tags(
            {
                "gate": "promotion",
                "decision": decision.decision,
                "candidate_version": candidate_version or "",
                "champion_version": champion_version or "",
                "candidate_run_id": out["run_id"],
            }
        )
        mlflow.log_params(
            {
                "primary_metric": policy.primary.name,
                "tolerance": policy.tolerance,
                "min_improvement": policy.min_improvement,
                "heldout": out["heldout"][:500],
                "reason": (args.reason or "")[:500],
            }
        )
        flat = {f"candidate_{k}": v for k, v in out["candidate"].items()}
        flat.update({f"champion_{k}": v for k, v in (out["champion"] or {}).items()})
        flat.update(
            {
                f"delta_{c['metric']}": c["delta"]
                for c in decision.comparison
                if c["delta"] is not None
            }
        )
        mlflow.log_metrics(
            {k: float(v) for k, v in flat.items() if v is not None and np.isfinite(v)}
        )
        mlflow.log_dict(record, "promotion/decision.json")
        mlflow.log_dict(
            {"candidate": out["candidate"], "champion": out["champion"]}, "promotion/metrics.json"
        )
        record["gate_run_id"] = run.info.run_id

    if not args.dry_run:
        engine = create_db_engine(get_settings("ml-trainer").postgres_url)
        record_promotion(
            engine, model, candidate_version, out["run_id"], record,
            {"candidate": out["candidate"], "champion": out["champion"]},
        )  # fmt: skip
        resolved = resolve_recommendation(
            engine,
            model,
            {
                "decision": decision.decision,
                "candidate_version": candidate_version,
                "gate_run_id": record["gate_run_id"],
                "reasons": decision.reasons,
            },
        )
        if resolved:
            log.info("%s: retraining recommendation resolved (%s)", model, decision.decision)
        engine.dispose()
        if decision.promoted:
            from ml.monitoring.reference import save_for_run

            save_for_run(model, candidate_version, out["run_id"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=float(os.getenv("PROMOTION_TOLERANCE", "0.01")),
        help="largest allowed regression on a guard metric (absolute; some guards have their own)",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=float(os.getenv("PROMOTION_MIN_IMPROVEMENT", "0.005")),
        help="how much better than the champion the primary metric must be",
    )
    parser.add_argument("--reason", default="manual", help="why this retraining runs (logged)")
    parser.add_argument("--dry-run", action="store_true", help="decide but don't register anything")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=int(os.getenv("TRAIN_THREADS", "6")))
    text = parser.add_argument_group("incident-text-classifier")
    text.add_argument("--candidate", choices=["distilbert", "tfidf"], default="distilbert")
    text.add_argument("--epochs", type=int, default=1)
    text.add_argument("--learning-rate", type=float, default=3e-5, help="warm-start learning rate")
    text.add_argument(
        "--from-scratch", action="store_true", help="don't warm-start from the champion"
    )
    text.add_argument("--recent-hours", type=float, default=2.0)
    text.add_argument("--recent-max", type=int, default=2000)
    text.add_argument("--recent-weight", type=int, default=2)
    args = parser.parse_args()
    record = run_gate(args)
    print("\n" + "=" * 72)
    print(f"{args.model}: candidate v{record['candidate_version']} vs champion "
          f"v{record['champion_version']}  ->  {record['decision'].upper()}")  # fmt: skip
    print(f"held-out: {record['heldout']}")
    for c in record["comparison"]:
        champ = "   n/a" if c["champion"] is None else f"{c['champion']:.4f}"
        cand = "   n/a" if c["candidate"] is None else f"{c['candidate']:.4f}"
        delta = "" if c["delta"] is None else f"{c['delta']:+.4f}"
        mark = "ok " if c["ok"] else "BAD"
        print(
            f"  {mark} {c['role']:<7} {c['metric']:<34} champion {champ}  candidate {cand}  {delta}"
        )
    for reason in record["reasons"]:
        print(f"  - {reason}")
    print(json.dumps({"gate_run_id": record.get("gate_run_id")}))


if __name__ == "__main__":
    main()
