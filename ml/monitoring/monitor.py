"""Runs the drift checks for the champion of each monitored model and records the results:

- rows in ``drift_reports`` (one per feature, plus an overall row with the full result);
- an HTML report under ``DRIFT_REPORTS_DIR/<model>/<check_id>.html`` (served by the API);
- a run in the MLflow experiment "drift-monitoring" with the metrics and the HTML report;
- a ``DriftAlert`` on ``drift.alerts`` when the model drifted, with the drifted features and
  their scores.

Drift alerts are consumed back (group ``ml-monitoring``) to record "retraining recommended" for
the model, or, with DRIFT_AUTO_RETRAIN=true, to run the retrain CLI (ml.lifecycle.retrain).
"""

import json
import logging
import subprocess
import sys
import tempfile
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from prometheus_client import Gauge, Histogram
from sqlalchemy import Engine, insert, text

from ml.lifecycle.records import recommend_retraining, set_recommendation_status
from ml.monitoring import data, reference
from ml.monitoring.config import MonitorConfig
from ml.monitoring.drift import CheckResult, run_check
from ml.monitoring.specs import SPECS
from traffic_common.db.models import DriftReport
from traffic_common.schemas import DriftAlert
from traffic_common.topics import Topic

log = logging.getLogger(__name__)
RETRAINABLE = ("congestion-classifier", "traffic-anomaly-detector", "incident-text-classifier")
FRAME_FILES_MINUTES = 10  # video_ingest keeps about this much of frame files
STATUS_VALUES = {"no_data": -1, "ok": 0, "warning": 1, "drifted": 2}
DRIFT_STATUS = Gauge(
    "traffic_drift_status",
    "Latest drift check per model: -1 no_data, 0 ok, 1 warning, 2 drifted",
    ["model"],
)
DRIFT_SHARE = Gauge(
    "traffic_drift_share", "Share of drifted features in the latest check", ["model"]
)
DRIFT_CHECK_SECONDS = Histogram(
    "traffic_drift_check_seconds",
    "Duration of one model's drift check",
    ["model"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)


class Monitor:
    def __init__(
        self,
        config: MonitorConfig,
        engine: Engine,
        *,
        bootstrap: str,
        producer=None,
        client=None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.bootstrap = bootstrap
        self.producer = producer
        self._client = client
        self._classifiers: dict[str, Any] = {}
        self._consecutive: dict[str, int] = {}
        self._lock = threading.Lock()  # one check at a time (scheduler and on-demand runs)
        self._retraining: dict[str, subprocess.Popen] = {}
        # When each model's HTML report was last logged to MLflow (window end), so an "ok"
        # report is logged at most every mlflow_html_every_minutes. Seeded from drift_reports.
        self._last_html: dict[str, datetime | None] = {}

    @property
    def client(self):
        if self._client is None:
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient()
        return self._client

    # ------------------------------------------------------------------ checks

    def run_all(self, **kwargs) -> list[CheckResult]:
        results = []
        for model in self.config.models:
            try:
                results.append(self.check(model, **kwargs))
            except Exception:
                log.exception("drift check for %s failed", model)
        return results

    def check(
        self,
        model_name: str,
        *,
        window_minutes: float | None = None,
        now: datetime | None = None,
        publish: bool = True,
        log_mlflow: bool = True,
        store: bool = True,
    ) -> CheckResult:
        """``store=False`` (a dry run) records nothing: no rows, MLflow run or alert."""
        with self._lock, DRIFT_CHECK_SECONDS.labels(model_name).time():
            result = self._check(model_name, window_minutes, now, publish, log_mlflow, store)
        if store:
            DRIFT_STATUS.labels(model_name).set(STATUS_VALUES.get(result.status, -1))
            DRIFT_SHARE.labels(model_name).set(result.drift_share)
        return result

    def _check(
        self, model_name, window_minutes, now, publish, log_mlflow, store=True
    ) -> CheckResult:
        spec = SPECS[model_name]
        version, run_id = reference.champion(model_name, self.client)
        ref = self.reference(model_name, version)
        end = now or datetime.now(UTC)
        default_window = (
            self.config.text_window_minutes
            if model_name == "incident-text-classifier"
            else self.config.window_minutes
        )
        start = end - timedelta(minutes=window_minutes or default_window)
        current, last_seen, rows = self.current(model_name, version, start, end, ref)
        result = run_check(
            spec,
            version,
            ref.frame,
            current,
            self.config,
            (start, end),
            reports_dir=self.config.reports_dir / model_name,
            last_seen=last_seen,
            reference_meta={k: v for k, v in ref.meta.items() if k != "columns"},
            current_rows=rows,
        )
        self._prune(self.config.reports_dir / model_name)
        if not store:
            return result
        consecutive = self._consecutive.get(model_name, 0) + 1 if result.status == "drifted" else 0
        self._consecutive[model_name] = consecutive
        mlflow_info = self._log_mlflow(result) if log_mlflow else {}
        alert = (
            self._alert(result, mlflow_info, consecutive) if result.status == "drifted" else None
        )
        self._store(result, mlflow_info, alert, consecutive)
        if alert is not None and publish and self.producer is not None:
            self.producer.send(Topic.DRIFT_ALERTS, alert)
            self.producer.flush(10)
        log.info(
            "drift %s v%s: %s (share %.2f, %d/%d rows) drifted=%s quality=%s",
            model_name,
            version,
            result.status.upper(),
            result.drift_share,
            result.current_rows,
            result.reference_rows,
            {f.feature: round(f.score, 3) for f in result.drifted_features if f.score is not None}
            or "none",
            [q.message for q in result.quality] or "ok",
        )
        return result

    def reference(self, model_name: str, version: str) -> reference.Reference:
        ref = reference.load(model_name, version, self.config.reference_root)
        if ref is not None:
            return ref
        if not self.config.auto_reference:
            raise RuntimeError(
                f"no reference for {model_name} v{version}; build it with "
                f"python -m ml.monitoring.reference build {model_name}"
            )
        log.info("no reference for %s v%s yet: building it", model_name, version)
        return reference.build(
            model_name,
            version,
            engine=self.engine,
            client=self.client,
            bootstrap=self.bootstrap,
            frames_dir=self.config.frames_dir,
            root=self.config.reference_root,
        )

    def current(
        self, model_name: str, version: str, start: datetime, end: datetime, ref
    ) -> tuple[pd.DataFrame, dict | None, int]:
        """Production data for the window, each camera's newest data (for staleness) and the
        number of rows that count for the minimum (frames, metric windows or reports)."""
        if model_name == "vehicle-detector":
            detections = data.read_detections(self.bootstrap, start, end)
            frame = data.detection_rows(detections, version)
            if frame.empty and detections:
                frame = data.detection_rows(detections)
            files_since = max(start, end - timedelta(minutes=FRAME_FILES_MINUTES))
            stats = data.frame_stats(self.config.frames_dir, files_since, until=end)
            frames = frame[frame["objects_per_frame"].notna()] if len(frame) else frame
            seen = data.last_seen(frames, data.camera_ids(self.engine))
            return pd.concat([frame, stats], ignore_index=True), seen, len(frames)
        if model_name in ("congestion-classifier", "traffic-anomaly-detector"):
            frame = data.traffic_rows(self.engine, start, end)
            seen = data.last_seen(frame, data.camera_ids(self.engine))
            return frame, seen, len(frame)
        frame = data.text_rows(self.engine, start, end).head(self.config.max_texts)
        if len(frame) and ref.arrays:
            from ml.monitoring.embeddings import TextEmbedder, project

            embedder = TextEmbedder(self._text_classifier(version))
            vectors = embedder.raw(frame["text"].astype(str).tolist())
            frame = pd.concat([frame.reset_index(drop=True), project(vectors, ref.arrays)], axis=1)
        return frame, None, len(frame)

    def _text_classifier(self, version: str):
        if version not in self._classifiers:
            import mlflow.pyfunc

            model = mlflow.pyfunc.load_model(f"models:/incident-text-classifier/{version}")
            self._classifiers = {version: model.unwrap_python_model()}
        return self._classifiers[version]

    def _prune(self, directory: Path) -> None:
        files = sorted(directory.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[self.config.reports_keep :]:
            old.unlink(missing_ok=True)

    # ------------------------------------------------------------------ outputs

    def _log_mlflow(self, result: CheckResult) -> dict[str, str]:
        from mlflow.entities import Metric, Param, RunTag

        try:
            client = self.client
            exp = client.get_experiment_by_name(self.config.experiment)
            exp_id = exp.experiment_id if exp else client.create_experiment(self.config.experiment)
            run = client.create_run(
                exp_id,
                run_name=f"{result.model_name}-{result.window_end:%Y%m%d-%H%M%S}",
                tags={
                    "model_name": result.model_name,
                    "model_version": result.model_version,
                    "status": result.status,
                    "check_id": str(result.check_id),
                },
            )
            run_id = run.info.run_id
            ts = int(result.window_end.timestamp() * 1000)
            metrics = [
                Metric("drift_share", result.drift_share, ts, 0),
                Metric("drifted_features", len(result.drifted_features), ts, 0),
                Metric("current_rows", result.current_rows, ts, 0),
                Metric("reference_rows", result.reference_rows, ts, 0),
                Metric("quality_issues", len(result.quality), ts, 0),
                *(
                    Metric(f"drift_score/{f.feature}", f.score, ts, 0)
                    for f in result.features
                    if f.score is not None
                ),
            ]
            params = [
                Param("window_start", result.window_start.isoformat()),
                Param("window_end", result.window_end.isoformat()),
                Param("share_threshold", str(result.share_threshold)),
                Param("reference_source", str(result.reference.get("source", ""))[:500]),
                *(Param(f"threshold/{f.feature}", str(f.threshold)) for f in result.features),
            ]
            drifted = ",".join(f.feature for f in result.drifted_features)
            client.log_batch(
                run_id, metrics=metrics, params=params, tags=[RunTag("drifted", drifted)]
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "result.json"
                path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
                client.log_artifact(run_id, str(path))
            html = result.html_path is not None and self.html_due(result)
            if html:
                client.log_artifact(run_id, str(result.html_path))
                self._last_html[result.model_name] = result.window_end
            client.set_tag(run_id, "html_report", "logged" if html else "on disk only")
            client.set_terminated(run_id)
            return {"run_id": run_id, "experiment_id": exp_id, "html": html}
        except Exception as exc:  # the check itself is still recorded
            log.warning("could not log drift check to MLflow: %s", exc)
            return {}

    def html_due(self, result: CheckResult) -> bool:
        """Log the (~4 MB) HTML report to MLflow for every warning or drifted check, and for
        an ok check only if the model's last logged report is mlflow_html_every_minutes old.
        Metrics are always logged; every report is also kept on disk (reports_keep)."""
        if result.status in ("warning", "drifted") or self.config.mlflow_html_every_minutes <= 0:
            return True
        if result.model_name not in self._last_html:
            self._last_html[result.model_name] = self._last_html_from_db(result.model_name)
        last = self._last_html[result.model_name]
        every = timedelta(minutes=self.config.mlflow_html_every_minutes)
        return last is None or result.window_end - last >= every

    def _last_html_from_db(self, model_name: str) -> datetime | None:
        try:
            with self.engine.connect() as conn:
                return conn.execute(
                    text(
                        "SELECT max(window_end) FROM drift_reports WHERE model_name = :m "
                        "AND feature IS NULL AND (report->'mlflow'->>'html')::boolean"
                    ),
                    {"m": model_name},
                ).scalar()
        except Exception as exc:
            log.warning("could not read when %s's report was last logged: %s", model_name, exc)
            return None

    def _alert(self, result: CheckResult, mlflow_info: dict, consecutive: int) -> DriftAlert:
        return DriftAlert(
            window_start=result.window_start,
            window_end=result.window_end,
            model_name=result.model_name,
            model_version=result.model_version,
            drift_type=result.drift_type,
            metric="drift_share",
            drift_score=result.drift_share,
            threshold=result.share_threshold,
            details={
                "check_id": str(result.check_id),
                "status": result.status,
                "drifted_features": [
                    {
                        "feature": f.feature,
                        "drift_type": f.drift_type,
                        "method": f.method,
                        "score": f.score,
                        "threshold": f.threshold,
                    }
                    for f in result.drifted_features
                ],
                "features_evaluated": sum(f.score is not None for f in result.features),
                "quality_issues": [q.message for q in result.quality],
                "report_url": f"/api/v1/drift/reports/{result.check_id}"
                if result.html_path
                else None,
                "mlflow_run_id": mlflow_info.get("run_id"),
                "consecutive_drifted_checks": consecutive,
            },
        )

    def _store(
        self,
        result: CheckResult,
        mlflow_info: dict,
        alert: DriftAlert | None,
        consecutive: int,
    ) -> None:
        window = {
            "window_start": result.window_start,
            "window_end": result.window_end,
            "check_id": result.check_id,
            "model_name": result.model_name,
            "model_version": result.model_version,
        }
        report = {
            **result.to_dict(),
            "report_url": f"/api/v1/drift/reports/{result.check_id}" if result.html_path else None,
            "mlflow": mlflow_info,
            "alert": alert.model_dump(mode="json") if alert else None,
            "consecutive_drifted_checks": consecutive,
        }
        rows = [
            {
                **window,
                "drift_type": f.drift_type,
                "metric": f.method,
                "feature": f.feature,
                "drift_score": f.score,
                "threshold": f.threshold,
                "is_drift": f.drifted,
                "report": {"reference": f.reference_summary, "current": f.current_summary},
            }
            for f in result.features
            if f.score is not None
        ]
        rows.append(
            {
                **window,
                "drift_type": result.drift_type,
                "metric": "drift_share",
                "feature": None,
                "drift_score": result.drift_share,
                "threshold": result.share_threshold,
                "is_drift": result.status == "drifted",
                "status": result.status,
                "report": report,
            }
        )
        with self.engine.begin() as conn:
            for row in rows:
                conn.execute(insert(DriftReport).values(**row))

    # ------------------------------------------------------------------ alerts -> retraining

    def on_alert(self, alert: DriftAlert) -> None:
        """Handler for drift.alerts: recommend retraining, or retrain with auto_retrain."""
        features = alert.details.get("drifted_features", [])
        names = ", ".join(f["feature"] for f in features) or "none listed"
        reason = (
            f"drift alert: {alert.drift_score:.0%} of features drifted "
            f"(threshold {alert.threshold:.0%}): {names}"
        )
        auto = self.config.auto_retrain and alert.model_name in RETRAINABLE
        row = recommend_retraining(
            self.engine,
            alert.model_name,
            alert.model_version,
            reason,
            features,
            str(alert.message_id),
            alert.timestamp,
        )
        if row is None:
            return  # already counted
        log.warning(
            "%s v%s: retraining %s (%d alert(s)): %s",
            alert.model_name,
            alert.model_version,
            "starting" if auto else "recommended",
            row["alert_count"],
            reason,
        )
        if auto:
            self._start_retrain(alert.model_name)

    def _start_retrain(self, model_name: str) -> None:
        running = self._retraining.get(model_name)
        if running is not None and running.poll() is None:
            log.info("%s: retraining already running", model_name)
            return
        set_recommendation_status(self.engine, model_name, "retraining")
        self._retraining[model_name] = subprocess.Popen(
            [sys.executable, "-m", "ml.lifecycle.retrain", model_name, "--reason", "drift alert"]
        )
