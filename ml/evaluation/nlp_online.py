"""Online evaluation of nlp_service: compare what it published (nlp_results) with the
simulator's ground truth (simulated_reports) for the last N hours.

    docker compose run --rm ml-trainer python -m ml.evaluation.nlp_online --hours 24

Per source style (driver, police, operator, social_media) and overall: incident type and
severity accuracy / macro-F1, camera accuracy (no camera counts as wrong), camera coverage
and precision, plus how many simulated reports have no result at all (lost or lagging).
Logged as a run in MLflow experiment "incident-text-classifier" (tag eval=online)."""

import argparse
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sqlalchemy import text

from traffic_common.db import create_db_engine

EXPERIMENT = "incident-text-classifier"

QUERY = text(
    """
    SELECT s.id AS report_id, s.style, s.stage, s.emitted_at, s.scenario_run_id,
           s.incident_type AS true_type, s.severity AS true_severity, s.camera_id AS true_camera,
           n.incident_type, n.severity, n.camera_id, n.incident_confidence, n.stage AS pred_stage,
           n.camera_confidence, n.model_name, n.model_version, n.processed_at
    FROM simulated_reports s
    LEFT JOIN LATERAL (
        SELECT * FROM nlp_results r
        WHERE r.report_id = s.id AND (CAST(:version AS text) IS NULL OR r.model_version = :version)
        ORDER BY r.processed_at DESC LIMIT 1
    ) n ON true
    WHERE s.emitted_at > now() - make_interval(secs => :seconds)
    """
)


def load(engine, hours: float, version: str | None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(QUERY, conn, params={"seconds": hours * 3600, "version": version})


def group_metrics(df: pd.DataFrame) -> dict[str, float]:
    done = df[df.incident_type.notna()]
    n = len(done)
    if not n:
        return {"n": 0, "unprocessed": len(df)}
    matched = done.camera_id.notna()
    correct_cam = done.camera_id == done.true_camera
    return {
        "n": n,
        "unprocessed": int(len(df) - n),
        "incident_accuracy": float((done.incident_type == done.true_type).mean()),
        "incident_macro_f1": float(f1_score(done.true_type, done.incident_type, average="macro")),
        "severity_accuracy": float((done.severity == done.true_severity).mean()),
        "severity_macro_f1": float(f1_score(done.true_severity, done.severity, average="macro")),
        "camera_accuracy": float(correct_cam.mean()),
        "camera_coverage": float(matched.mean()),
        "camera_precision": float(correct_cam[matched].mean()) if matched.any() else np.nan,
        "mean_incident_confidence": float(done.incident_confidence.mean()),
        # Clearing detection ("cleared" reports speed up incident resolution in fusion).
        "cleared_recall": _ratio(
            ((done.pred_stage == "cleared") & (done.stage == "cleared")).sum(),
            (done.stage == "cleared").sum(),
        ),
        "cleared_precision": _ratio(
            ((done.pred_stage == "cleared") & (done.stage == "cleared")).sum(),
            (done.pred_stage == "cleared").sum(),
        ),
    }


def _ratio(a, b) -> float:
    return float(a) / float(b) if b else np.nan


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"style": "all", **group_metrics(df)}]
    for style, g in df.groupby("style"):
        rows.append({"style": style, **group_metrics(g)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--model-version", default=None, help="only results of this version")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    engine = create_db_engine(os.getenv("POSTGRES_URL"))
    df = load(engine, args.hours, args.model_version)
    if df.empty:
        raise SystemExit(f"no simulated reports in the last {args.hours:g} h")
    table = evaluate(df)
    versions = sorted(df.model_version.dropna().unique().tolist())
    pd.set_option("display.width", 200)
    print(f"last {args.hours:g} h: {len(df)} simulated reports, model versions {versions}\n")
    print(table.round(3).to_string(index=False))

    if args.no_mlflow:
        return
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"online-eval-{datetime.now(UTC):%Y%m%d-%H%M%S}"):
        mlflow.set_tags({"eval": "online", "model_versions": ",".join(versions) or "-"})
        mlflow.log_params({"hours": args.hours, "model_version_filter": args.model_version})
        for row in table.to_dict("records"):
            style = row.pop("style")
            mlflow.log_metrics(
                {f"online_{style}_{k}": float(v) for k, v in row.items() if pd.notna(v)}
            )
        out = Path(tempfile.mkdtemp(prefix="nlp-online-"))
        table.to_csv(out / "online_by_style.csv", index=False)
        wrong = df[df.incident_type.notna() & (df.incident_type != df.true_type)]
        wrong.to_csv(out / "online_misclassified.csv", index=False)
        mlflow.log_artifacts(str(out), artifact_path="online")


if __name__ == "__main__":
    main()
