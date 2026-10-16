"""Consume reports.text in batches: store each report, analyse it, store the result in
nlp_results and publish it to nlp.results."""

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert

from ml.nlp.matching import CameraMatcher
from ml.registry import LoadedModel, ModelManager
from nlp_service.config import NlpConfig
from nlp_service.pipeline import NlpPipeline
from traffic_common.db.models import Camera, NlpResult, Report
from traffic_common.db.registry import record_model_version
from traffic_common.gazetteer import CameraInfo
from traffic_common.schemas import NLPResult, TextReport
from traffic_common.topics import Topic

log = logging.getLogger(__name__)


@dataclass
class Stats:
    reports: int = 0
    batches: int = 0
    largest_batch: int = 0
    skipped_duplicates: int = 0
    published: int = 0
    camera_matched: int = 0
    seconds: float = 0.0
    types: dict[str, int] = field(default_factory=dict)

    def reset(self) -> str:
        per = self.seconds * 1000 / self.reports if self.reports else 0.0
        text = (
            f"reports={self.reports} batches={self.batches} largest_batch={self.largest_batch} "
            f"duplicates={self.skipped_duplicates} published={self.published} "
            f"camera_matched={self.camera_matched} ms_per_report={per:.1f} types={self.types}"
        )
        self.__init__()  # type: ignore[misc]
        return text


def load_cameras(engine: Engine) -> list[CameraInfo]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(Camera.id, Camera.name, Camera.location_name, Camera.latitude, Camera.longitude)
        ).all()
    return [CameraInfo(*r) for r in rows]


class NlpService:
    def __init__(
        self,
        config: NlpConfig,
        engine: Engine,
        models: ModelManager,
        send,
        ner=None,
    ) -> None:
        self.cfg = config
        self.engine = engine
        self.models = models
        self.send = send  # (topic, message) -> None
        self.stats = Stats()
        self._lock = threading.Lock()
        self._matcher: CameraMatcher | None = None
        self.camera_ids: set[str] = set()
        self.refresh_cameras()
        self.pipeline = NlpPipeline(lambda: self.models.get("classifier"), self.matcher, ner)

    # ------------------------------------------------------------------ cameras

    def matcher(self) -> CameraMatcher:
        with self._lock:
            assert self._matcher is not None
            return self._matcher

    def refresh_cameras(self) -> None:
        cameras = load_cameras(self.engine)
        ids = {c.id for c in cameras}
        if ids == self.camera_ids and self._matcher is not None:
            return
        matcher = CameraMatcher(cameras, self.cfg.match_min_score, self.cfg.match_min_confidence)
        with self._lock:
            self._matcher, self.camera_ids = matcher, ids
        log.info("camera matcher: %d cameras, %d aliases", len(cameras), len(matcher.aliases))

    # ------------------------------------------------------------------ messages

    def handle(self, report: TextReport) -> None:
        self.handle_batch([report])

    def handle_batch(self, reports: list[TextReport]) -> None:
        """Consumer batch handler. Idempotent: a report already analysed by the current
        model version (redelivery after a crash) is skipped."""
        unique = list({r.report_id: r for r in reports}.values())
        started = datetime.now(UTC)
        self._store_reports(unique)
        model = self.models.get("classifier")
        done = self._already_analysed([r.report_id for r in unique], model.version)
        todo = [r for r in unique if r.report_id not in done]
        self.stats.skipped_duplicates += len(reports) - len(todo)
        results = self.pipeline.analyse(todo)
        self._store_results(results)
        for result in results:
            self.send(Topic.NLP_RESULTS, result)
            self.stats.published += 1
            self.stats.camera_matched += result.camera_id is not None
            key = str(result.incident_type)
            self.stats.types[key] = self.stats.types.get(key, 0) + 1
        self.stats.reports += len(todo)
        self.stats.batches += 1
        self.stats.largest_batch = max(self.stats.largest_batch, len(reports))
        self.stats.seconds += (datetime.now(UTC) - started).total_seconds()
        if len(reports) > 1:
            log.debug("batch of %d reports analysed", len(reports))

    # ------------------------------------------------------------------ persistence

    def _store_reports(self, reports: list[TextReport]) -> None:
        rows = [
            {
                "id": r.report_id,
                "source": r.source,
                "text": r.text,
                # Only cameras the database knows (the column is a foreign key).
                "camera_id": r.camera_id if r.camera_id in self.camera_ids else None,
                "location_name": r.location_name,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "reported_at": r.timestamp,
            }
            for r in reports
        ]
        if rows:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(Report).values(rows).on_conflict_do_nothing(index_elements=["id"])
                )

    def _already_analysed(self, report_ids: list, model_version: str) -> set:
        if not report_ids:
            return set()
        with self.engine.connect() as conn:
            return set(
                conn.scalars(
                    select(NlpResult.report_id).where(
                        NlpResult.report_id.in_(report_ids),
                        NlpResult.model_version == model_version,
                    )
                )
            )

    def _store_results(self, results: list[NLPResult]) -> None:
        if not results:
            return
        rows = [
            {
                "message_id": r.message_id,
                "report_id": r.report_id,
                "camera_id": r.camera_id if r.camera_id in self.camera_ids else None,
                "camera_confidence": r.camera_confidence,
                "incident_type": r.incident_type,
                "incident_confidence": r.incident_confidence,
                "severity": r.severity,
                "severity_confidence": r.severity_confidence,
                "probabilities": r.probabilities,
                "stage": r.stage,
                "sentiment": r.sentiment,
                "location_name": r.location_name,
                "entities": [e.model_dump() for e in r.entities],
                "model_name": r.model_name,
                "model_version": r.model_version,
                "model_versions": r.model_versions,
                "details": r.details,
                "processed_at": r.timestamp,
            }
            for r in results
        ]
        with self.engine.begin() as conn:
            conn.execute(
                insert(NlpResult).values(rows).on_conflict_do_nothing(index_elements=["message_id"])
            )

    # ------------------------------------------------------------------ models and stats

    def record_model(self, model: LoadedModel) -> None:
        if model.source != "mlflow":
            return
        try:
            record_model_version(
                self.engine,
                model.name,
                model.version,
                mlflow_run_id=model.run_id,
                metrics={"role": model.role, "loaded_by": "nlp_service"},
            )
        except Exception as exc:
            log.warning("could not record %s in model_versions: %s", model.label, exc)

    def log_stats(self) -> None:
        log.info(
            "stats %s model=%s cameras=%d",
            self.stats.reset(),
            self.models.get("classifier").label,
            len(self.camera_ids),
        )
