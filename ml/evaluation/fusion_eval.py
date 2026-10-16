"""Evaluate fused incidents against the simulator's ground truth (scenario_runs).

    docker compose run --rm ml-trainer python -m ml.evaluation.fusion_eval --hours 2

For every finished scenario run in the period, its detection is the first incident on its
camera suspected between the scenario start (minus a minute) and its end, or else an
incident already open then that the scenario's evidence joined (``already_open``):
- time from scenario start to suspected and to confirmed (wall clock; scenario time is
  this x time_scale)
- incident type and severity (vs the scenario's peak severity) correct or not
- time to resolve: incident end minus the start of the scenario's last (clearing) phase
- fragments: further incidents on the camera during the scenario

False incidents: incidents not matched to a scenario and suspected outside every scenario
window (start - 1 min to end + ``--grace`` minutes) on that camera, per camera-hour outside
scenarios. Suspected and
confirmed are counted separately. Results go to MLflow experiment "incident-fusion".
"""

import argparse
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from traffic_common.db import create_db_engine

EXPERIMENT = "incident-fusion"
LEAD = timedelta(minutes=1)


def load(engine, since: datetime, until: datetime) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    with engine.connect() as conn:
        runs = pd.read_sql(
            text(
                "SELECT id, scenario_name, camera_id, incident_type, peak_severity, time_scale, "
                "started_at, coalesce(ended_at, expected_end_at) AS ended_at, timeline, status "
                "FROM scenario_runs WHERE started_at >= :since AND started_at < :until "
                "AND status <> 'running' ORDER BY started_at"
            ),
            conn,
            params={"since": since, "until": until},
        )
        incidents = pd.read_sql(
            text(
                "SELECT id, camera_id, status, incident_type, severity, fused_confidence, "
                "start_time, suspected_at, confirmed_at, end_time, details->'at_confirmation' "
                "AS at_confirmation, details->>'resolution' AS resolution FROM incidents "
                "WHERE suspected_at >= :since AND suspected_at < :until ORDER BY suspected_at"
            ),
            conn,
            params={"since": since - timedelta(hours=1), "until": until},
        )
        cameras = conn.scalar(text("SELECT count(*) FROM cameras"))
    for frame, cols in (
        (runs, ["started_at", "ended_at"]),
        (incidents, ["start_time", "suspected_at", "confirmed_at", "end_time"]),
    ):
        for col in cols:
            frame[col] = pd.to_datetime(frame[col], utc=True)
    return runs, incidents, int(cameras)


def _seconds(t, start) -> float:
    return (t - start).total_seconds() if pd.notna(t) else np.nan


def match_runs(runs: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in runs.itertuples():
        on_cam = incidents[incidents.camera_id == run.camera_id]
        # Incidents open at some point during the scenario. Prefer one suspected after it
        # started; otherwise an incident already open (e.g. from a false alarm just before)
        # that the scenario's evidence joined counts too, with a negative time to suspected.
        overlap = on_cam[
            (on_cam.suspected_at <= run.ended_at)
            & (on_cam.end_time.isna() | (on_cam.end_time >= run.started_at))
        ]
        during = overlap[overlap.suspected_at >= run.started_at - LEAD]
        if during.empty and not overlap.empty:
            during = overlap.tail(1)
        clearing_start = pd.Timestamp(run.timeline[-1]["starts_at"]) if run.timeline else None
        row = {
            "scenario_run_id": run.id,
            "scenario": run.scenario_name,
            "camera_id": run.camera_id,
            "time_scale": run.time_scale,
            "started_at": run.started_at,
            "true_type": run.incident_type,
            "true_severity": run.peak_severity,
            "detected": not during.empty,
            "fragments": max(len(overlap) - 1, 0),
        }
        if not during.empty:
            inc = during.iloc[0]
            row.update(
                incident_id=inc.id,
                already_open=bool(inc.suspected_at < run.started_at - LEAD),
                status=inc.status,
                type=inc.incident_type,
                severity=inc.severity,
                type_correct=inc.incident_type == run.incident_type,
                severity_correct=inc.severity == run.peak_severity,
                to_suspected_s=_seconds(inc.suspected_at, run.started_at),
                to_confirmed_s=_seconds(inc.confirmed_at, run.started_at),
                resolve_after_clearing_s=(inc.end_time - clearing_start).total_seconds()
                if pd.notna(inc.end_time) and clearing_start is not None
                else np.nan,
                resolve_after_scenario_end_s=(inc.end_time - run.ended_at).total_seconds()
                if pd.notna(inc.end_time)
                else np.nan,
                confidence_at_confirmation=(inc.at_confirmation or {}).get("fused"),
                confirmation_rule=(inc.at_confirmation or {}).get("rule"),
                resolution=inc.resolution,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def false_incidents(
    runs: pd.DataFrame,
    incidents: pd.DataFrame,
    cameras: int,
    since: datetime,
    until: datetime,
    grace: timedelta,
    matched_ids: set = frozenset(),
) -> tuple[dict[str, float], pd.DataFrame]:
    def in_scenario(inc) -> bool:
        cam = runs[runs.camera_id == inc.camera_id]
        return bool(
            (
                (cam.started_at - LEAD <= inc.suspected_at)
                & (inc.suspected_at <= cam.ended_at + grace)
            ).any()
        )

    period = incidents[(incidents.suspected_at >= since) & (incidents.suspected_at < until)]
    false = period[[not in_scenario(i) and i.id not in matched_ids for i in period.itertuples()]]
    covered = sum(
        (
            min(r.ended_at + grace, pd.Timestamp(until))
            - max(r.started_at - LEAD, pd.Timestamp(since))
        ).total_seconds()
        for r in runs.itertuples()
    )
    camera_hours = max(cameras * (until - since).total_seconds() - covered, 1.0) / 3600
    confirmed = false[false.confirmed_at.notna()]
    metrics = {
        "camera_hours_outside_scenarios": camera_hours,
        "false_incidents": len(false),
        "false_confirmed": len(confirmed),
        "false_incidents_per_camera_hour": len(false) / camera_hours,
        "false_confirmed_per_camera_hour": len(confirmed) / camera_hours,
    }
    return metrics, false


def summarise(matched: pd.DataFrame) -> pd.DataFrame:
    if matched.empty:
        return pd.DataFrame()
    agg = {"runs": ("detected", "size"), "detection_rate": ("detected", "mean")}
    for col in ("to_suspected_s", "to_confirmed_s", "resolve_after_clearing_s"):
        if col in matched:
            agg[f"mean_{col}"] = (col, "mean")
    for col in ("type_correct", "severity_correct"):
        if col in matched:
            agg[f"{col.replace('_correct', '')}_accuracy"] = (col, "mean")
    return (
        matched.assign(
            type_correct=matched.get("type_correct", pd.Series(dtype=float)).astype(float),
            severity_correct=matched.get("severity_correct", pd.Series(dtype=float)).astype(float),
        )
        .groupby("scenario")
        .agg(**agg)
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--since", default=None, help="ISO start (overrides --hours)")
    parser.add_argument("--grace", type=float, default=10.0, help="minutes after a scenario")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    until = datetime.now(UTC)
    since = (
        datetime.fromisoformat(args.since) if args.since else until - timedelta(hours=args.hours)
    )
    engine = create_db_engine(os.getenv("POSTGRES_URL"))
    runs, incidents, cameras = load(engine, since, until)
    matched = match_runs(runs, incidents)
    summary = summarise(matched)
    false_metrics, false = false_incidents(
        runs,
        incidents,
        cameras,
        since,
        until,
        timedelta(minutes=args.grace),
        set(matched["incident_id"].dropna()) if "incident_id" in matched else set(),
    )
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(
        f"{since:%Y-%m-%d %H:%M} -> {until:%H:%M} UTC: {len(runs)} scenario runs, "
        f"{len(incidents)} incidents, {cameras} cameras\n"
    )
    if not matched.empty:
        print(
            matched.drop(columns=["scenario_run_id", "incident_id"], errors="ignore")
            .round(1)
            .to_string(index=False),
            "\n",
        )
        print(summary.round(3).to_string(index=False), "\n")
    print({k: round(v, 3) for k, v in false_metrics.items()})

    if args.no_mlflow:
        return
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"fusion-eval-{until:%Y%m%d-%H%M%S}"):
        mlflow.log_params(
            {
                "since": since.isoformat(),
                "until": until.isoformat(),
                "grace_minutes": args.grace,
                "scenario_runs": len(runs),
            }
        )
        metrics = dict(false_metrics)
        if not matched.empty:
            metrics["detection_rate"] = float(matched.detected.mean())
        for row in summary.to_dict("records"):
            name = row.pop("scenario")
            metrics.update({f"{name}_{k}": float(v) for k, v in row.items() if pd.notna(v)})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
        out = Path(tempfile.mkdtemp(prefix="fusion-eval-"))
        matched.to_csv(out / "scenario_runs.csv", index=False)
        summary.to_csv(out / "by_scenario.csv", index=False)
        false.to_csv(out / "false_incidents.csv", index=False)
        mlflow.log_artifacts(str(out), artifact_path="evaluation")


if __name__ == "__main__":
    main()
