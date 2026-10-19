"""Debounce incident versions, build the fact context, summarise (model or template), store the
summary and publish it."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

from prometheus_client import Counter
from sqlalchemy import Engine, text
from sqlalchemy.dialects.postgresql import insert

from ml.registry import LoadedModel, ModelManager
from ml.summarize.context import load_context
from summarizer_service.config import SummarizerServiceConfig
from traffic_common.db.models import Summary as SummaryRow
from traffic_common.db.registry import record_model_version
from traffic_common.schemas import Incident, Summary
from traffic_common.telemetry import INFERENCE_SECONDS, SERVICE_ERRORS
from traffic_common.topics import Topic

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 5
SUMMARIES = Counter(
    "traffic_summaries_total",
    "Summaries written, by path: model output, or the template (fallback or champion)",
    ["generation"],
)


@dataclass
class Pending:
    incident: dict  # the latest Incident version's fields
    due: float  # monotonic time
    reasons: set[str] = field(default_factory=set)
    attempts: int = 0


@dataclass
class Stats:
    received: int = 0
    debounced: int = 0
    summarised: int = 0
    model: int = 0
    template: int = 0
    failed: int = 0

    def reset(self) -> str:
        text_ = " ".join(f"{k}={v}" for k, v in vars(self).items())
        self.__init__()  # type: ignore[misc]
        return text_


def state_of(incident: dict) -> tuple[str, str, str]:
    return (str(incident["status"]), str(incident["incident_type"]), str(incident["severity"]))


def change_reasons(last: tuple | None, now: tuple) -> set[str]:
    if last is None:
        return {"created"}
    names = ("status", "type", "severity")
    return {f"{n} {a} -> {b}" for n, a, b in zip(names, last, now, strict=True) if a != b}


class SummarizerService:
    def __init__(
        self,
        config: SummarizerServiceConfig,
        engine: Engine,
        models: ModelManager,
        send: Callable,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = config
        self.engine = engine
        self.models = models
        self.send = send  # (topic, message) -> None
        self.monotonic = monotonic
        self.pending: dict[str, Pending] = {}
        self.last: dict[str, tuple] = {}  # last summarised (status, type, severity)
        self.stats = Stats()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ input and debounce

    def handle(self, message: Incident) -> None:
        self.stats.received += 1
        self.offer(message.model_dump(mode="python"))

    def offer(self, incident: dict) -> None:
        """Queue an incident version if it (or the pending one) is worth a summary; a newer
        version replaces the pending one and restarts the debounce timer."""
        key = str(incident["incident_id"])
        with self._lock:
            pending = self.pending.get(key)
            reasons = change_reasons(self.last.get(key), state_of(incident))
            if pending is not None:  # also what changed since the version already waiting
                reasons |= change_reasons(state_of(pending.incident), state_of(incident))
            if pending is None and not reasons:
                return  # only the confidence moved: no new summary
            if pending is not None:
                self.stats.debounced += 1
                if incident["version"] >= pending.incident["version"]:
                    pending.incident = incident
                pending.reasons |= reasons
                pending.due = self.monotonic() + self.cfg.debounce_seconds
            else:
                self.pending[key] = Pending(
                    incident, self.monotonic() + self.cfg.debounce_seconds, reasons
                )

    def due(self) -> list[Pending]:
        now = self.monotonic()
        with self._lock:
            ready = [k for k, p in self.pending.items() if p.due <= now]
            return [self.pending.pop(k) for k in ready]

    def tick(self) -> None:
        for p in self.due():
            try:
                self.summarise(p)
            except Exception as exc:
                p.attempts += 1
                if p.attempts >= MAX_ATTEMPTS:
                    self.stats.failed += 1
                    SERVICE_ERRORS.labels("summary").inc()
                    if isinstance(exc, LookupError):  # e.g. another database's test incident
                        log.warning("skipping incident %s: %s", p.incident["incident_id"], exc)
                    else:
                        log.exception("giving up on incident %s", p.incident["incident_id"])
                    continue
                log.warning("summary for %s failed (%s); retrying", p.incident["incident_id"], exc)
                with self._lock:
                    key = str(p.incident["incident_id"])
                    if key not in self.pending:  # a newer version would supersede it
                        p.due = self.monotonic() + 2.0 * p.attempts
                        self.pending[key] = p

    # ------------------------------------------------------------------ summarising

    def summarise(self, p: Pending) -> Summary:
        incident = p.incident
        with self.engine.connect() as conn:
            ctx = load_context(conn, incident["incident_id"], overrides=incident)
        if ctx is None:
            raise LookupError(f"incident {incident['incident_id']} not in Postgres yet")
        model = self.models.get("summarizer")
        result = model.predictor.summarizer.summarize(ctx, self.cfg.timeout_seconds)
        faith = {
            **result.faithfulness,
            "triggers": sorted(p.reasons),
            "configuration": f"{model.name}/{model.version}",
        }
        end = max(ctx.reference_time, ctx.start_time + timedelta(seconds=1))
        summary = Summary(
            scope="incident",
            camera_id=ctx.camera_id,
            incident_id=incident["incident_id"],
            incident_version=ctx.version,
            period_start=ctx.start_time,
            period_end=end,
            headline=result.headline[:200],
            text=result.summary,
            generation=result.generation,
            model_name=result.model_name,
            model_version=result.model_version,
            prompt_version=result.prompt_version,
            latency_ms=result.latency_ms,
            faithfulness=faith,
        )
        with self.engine.begin() as conn:
            conn.execute(
                insert(SummaryRow).values(
                    id=summary.summary_id,
                    scope="incident",
                    camera_id=summary.camera_id,
                    incident_id=summary.incident_id,
                    incident_version=summary.incident_version,
                    period_start=summary.period_start,
                    period_end=summary.period_end,
                    headline=summary.headline,
                    text=summary.text,
                    generation=summary.generation,
                    model_name=summary.model_name,
                    model_version=summary.model_version,
                    prompt_version=summary.prompt_version,
                    latency_ms=summary.latency_ms,
                    faithfulness=faith,
                    context=ctx.to_dict(),
                )
            )
        self.send(Topic.SUMMARIES, summary)
        key = str(incident["incident_id"])
        with self._lock:
            self.last[key] = state_of(incident)
        self.stats.summarised += 1
        setattr(self.stats, result.generation, getattr(self.stats, result.generation) + 1)
        SUMMARIES.labels(result.generation).inc()
        # The whole generation (with the faithfulness guard and any fallback), per summary.
        INFERENCE_SECONDS.labels(model.name).observe((result.latency_ms or 0) / 1000)
        log.warning(
            "SUMMARY v%d %s %s/%s via %s (%.0f ms%s): %s | %s",
            ctx.version,
            ctx.camera_id,
            ctx.status,
            ctx.incident_type,
            result.generation,
            result.latency_ms,
            f", {result.faithfulness.get('fallback_reason')}"
            if result.generation == "template"
            else "",
            summary.headline,
            summary.text,
        )
        return summary

    # ------------------------------------------------------------------ startup

    def recover(self) -> int:
        """Remember what was already summarised, and queue incident versions from the last
        ``catch_up_hours`` whose latest version has no summary (lost on a restart)."""
        with self.engine.connect() as conn:
            done = conn.execute(
                text(
                    "SELECT DISTINCT ON (incident_id) incident_id, context->>'status' AS status, "
                    "context->>'incident_type' AS t, context->>'severity' AS s FROM summaries "
                    "WHERE incident_id IS NOT NULL ORDER BY incident_id, created_at DESC"
                )
            ).all()
            missing = (
                conn.execute(
                    text(
                        "SELECT i.id, i.version, i.status, i.incident_type, i.severity, "
                        "i.fused_confidence, i.confidence_components, i.vision_degraded, "
                        "i.start_time, i.end_time, i.updated_at FROM incidents i "
                        "WHERE i.updated_at > now() - make_interval(secs => :s) "
                        "AND NOT EXISTS (SELECT 1 FROM summaries s "
                        "WHERE s.incident_id = i.id AND s.incident_version = i.version)"
                    ),
                    {"s": self.cfg.catch_up_hours * 3600},
                )
                .mappings()
                .all()
            )
        with self._lock:
            for r in done:
                self.last[str(r.incident_id)] = (r.status, r.t, r.s)
        queued = 0
        for r in missing:
            incident = {
                "incident_id": r["id"],
                "version": r["version"],
                "status": r["status"],
                "incident_type": r["incident_type"],
                "severity": r["severity"],
                "fused_confidence": r["fused_confidence"],
                "confidence_components": r["confidence_components"],
                "vision_degraded": r["vision_degraded"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "timestamp": r["updated_at"],
            }
            before = len(self.pending)
            self.offer(incident)
            queued += len(self.pending) > before
        if queued:
            log.info("queued %d incident version(s) without a summary", queued)
        return queued

    # ------------------------------------------------------------------ models and stats

    def record_model(self, model: LoadedModel) -> None:
        try:
            record_model_version(
                self.engine,
                model.name,
                model.version,
                mlflow_run_id=model.run_id,
                metrics={
                    "role": "summarizer",
                    "loaded_by": "summarizer",
                    "source": model.source,
                    "config": model.predictor.config.to_dict(),
                },
            )
        except Exception as exc:
            log.warning("could not record %s in model_versions: %s", model.label, exc)

    def log_stats(self) -> None:
        with self._lock:
            pending = len(self.pending)
        log.info(
            "stats %s pending=%d model=%s",
            self.stats.reset(),
            pending,
            self.models.get("summarizer").label,
        )
