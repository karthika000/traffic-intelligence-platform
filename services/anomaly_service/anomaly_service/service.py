"""Consume traffic.metrics and events.raw, score every few seconds, publish anomaly episodes."""

import copy
import json
import logging
import math
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
from prometheus_client import Counter, Gauge
from sqlalchemy import Engine, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from anomaly_service.calibration import CameraSettings, calibrate_camera, python_model
from anomaly_service.config import AnomalyConfig
from anomaly_service.models import LoadedModel, ModelManager
from anomaly_service.state import (
    HEALTH_RECENT_WINDOWS,
    ActiveAnomaly,
    CameraState,
    SystemEpisode,
    sensor_point,
)
from ml.alarm import AlarmEvent, AlarmTracker
from ml.features import SEQUENCE_LENGTH, WINDOW_S, current_baselines
from traffic_common.db.models import Anomaly as AnomalyRow
from traffic_common.db.models import Camera, CameraCongestion, CongestionHistory, SensorReading
from traffic_common.db.registry import record_model_version
from traffic_common.schemas import Anomaly, KafkaMessage, RawEvent, TrafficMetrics
from traffic_common.telemetry import INFERENCE_SECONDS, SERVICE_ERRORS, timed
from traffic_common.topics import Topic

log = logging.getLogger(__name__)
SEVERITY_ORDER = ["low", "medium", "high", "critical"]
# Cameras whose latest window is within this of the newest one count as "the same check".
SYSTEM_FRESH_S = 3 * WINDOW_S
SYSTEM_MODEL = ("system-health-rules", "1")
# Model inputs that measure the camera pipeline rather than traffic. Their normal error scale is
# tiny (cv_inference almost always processes the same number of frames per window), so a
# one-frame wobble can top the score; they only count when the health check sees a real drop.
HEALTH_FEATURES = {"frames_rel", "mean_confidence", "camera_speed_present"}
# Hints about traffic (not equipment): alerts with one of these can end by recovery.
TRAFFIC_HINTS = {"speed_drop", "stopped_vehicles", "pedestrian_in_road"}

ANOMALIES_ACTIVE = Gauge(
    "traffic_anomalies_active",
    "Active alerts: 1 per camera with one, plus the system scope",
    ["scope"],
)
CAMERAS_WARMING = Gauge(
    "traffic_cameras_warming_up",
    "Cameras that can't raise traffic alerts yet: in their warm-up, or without recent metrics",
)
ANOMALY_EVENTS = Counter(
    "traffic_anomaly_events_total",
    "Published anomaly events (start, update, end)",
    ["scope", "event"],
)


@dataclass
class Stats:
    metrics: int = 0
    stale_metrics: int = 0
    sensor_stored: int = 0
    scored: int = 0
    warming: int = 0  # checks of cameras still warming up (no alerts)
    started: int = 0
    updated: int = 0
    ended: int = 0
    recovered: int = 0  # of those, ended because the traffic metrics recovered
    health_noise: int = 0  # checks over threshold only through healthy pipeline features
    db_errors: int = 0

    def reset(self) -> str:
        text = " ".join(f"{k}={v}" for k, v in vars(self).items())
        self.__init__()  # type: ignore[misc]
        return text


class AnomalyService:
    def __init__(self, config: AnomalyConfig, engine: Engine, models: ModelManager, send) -> None:
        self.cfg = config
        self.engine = engine
        self.models = models
        self.send = send  # (topic, message) -> None
        self.cameras: dict[str, CameraState] = {}
        self._lock = threading.Lock()
        self._db_warned: set[str] = set()
        self.stats = Stats()
        self.system: SystemEpisode | None = None  # active system-wide degradation
        self.settings: dict[str, CameraSettings] = {}  # from cameras.zone_config["anomaly"]
        self._pristine: dict[str, object] = {}  # model label -> its calibration as loaded

    # ------------------------------------------------------------------ ingestion

    def camera(self, camera_id: str) -> CameraState:
        with self._lock:
            state = self.cameras.get(camera_id)
            if state is None:
                tracker = AlarmTracker(
                    self.cfg.cooldown_seconds,
                    self.cfg.clear_ratio,
                    self.cfg.clear_after,
                    self.cfg.min_consecutive,
                )
                state = CameraState(camera_id, self.cfg.history_windows, tracker)
                state.settings = self.settings.get(camera_id, CameraSettings())
                self.cameras[camera_id] = state
            return state

    def handle(self, message: KafkaMessage) -> None:
        """Consumer handler for both topics."""
        if isinstance(message, RawEvent):
            self._handle_event(message)
        elif isinstance(message, TrafficMetrics):
            self._handle_metrics(message)

    def _handle_event(self, event: RawEvent) -> None:
        point = sensor_point(event)
        if point is None or event.camera_id is None:
            return
        self._store_sensor(event)
        state = self.camera(event.camera_id)
        with state.lock:
            state.add_sensor(point)

    def _handle_metrics(self, m: TrafficMetrics) -> None:
        self.stats.metrics += 1
        if (datetime.now(UTC) - m.window_end).total_seconds() > self.cfg.max_metric_age_seconds:
            self.stats.stale_metrics += 1  # e.g. history replayed after a restart
            return
        state = self.camera(m.camera_id)
        with state.lock:
            latest = state.latest_ts
            gap = (m.window_end - latest).total_seconds() if latest is not None else 0.0
            if self.cfg.warmup_seconds > 0 and gap > self.cfg.warmup_seconds:
                log.info(
                    "camera %s: no metrics for %s; warming up again",
                    m.camera_id,
                    m.window_end - latest,
                )
                state.restart_warmup(m.window_end)
            state.add_metrics(m)

    def refresh_settings(self) -> None:
        """Re-read per-camera anomaly settings (cameras.zone_config["anomaly"]); a camera whose
        settings changed is recalibrated at its next check."""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(select(Camera.id, Camera.zone_config)).all()
        except Exception as exc:
            log.warning("could not read camera settings: %s", exc)
            return
        settings = {r.id: CameraSettings.from_zone_config(r.zone_config) for r in rows}
        for camera_id, new in settings.items():
            if self.settings.get(camera_id) != new and new != CameraSettings():
                log.info("camera %s: anomaly settings %s", camera_id, new)
        self.settings = settings
        with self._lock:
            states = list(self.cameras.values())
        for state in states:
            with state.lock:
                state.settings = settings.get(state.camera_id, CameraSettings())

    def _store_sensor(self, event: RawEvent) -> None:
        p = event.payload
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(SensorReading)
                    .values(
                        event_id=event.event_id,
                        camera_id=event.camera_id,
                        detector_id=p.get("detector_id"),
                        reading_at=event.timestamp,
                        vehicle_count=p.get("vehicle_count"),
                        flow_veh_per_hour=p.get("flow_veh_per_hour"),
                        avg_speed_kmh=p.get("avg_speed_kmh"),
                        occupancy=p.get("occupancy"),
                        interval_seconds=p.get("interval_seconds"),
                        scenario_id=event.scenario_id,
                    )
                    .on_conflict_do_nothing(index_elements=[SensorReading.event_id])
                )
            self.stats.sensor_stored += 1
        except IntegrityError:
            self._warn_once(event.camera_id, "sensor reading for a camera not in the cameras table")
        except Exception as exc:
            self.stats.db_errors += 1
            SERVICE_ERRORS.labels("database").inc()
            log.error("could not store sensor reading: %s", exc)

    # ------------------------------------------------------------------ scoring

    def score_all(self) -> None:
        with self._lock:
            states = list(self.cameras.values())
        congestion, anomaly = self.models.get("congestion"), self.models.get("anomaly")
        batch = []
        for state in states:
            with state.lock:
                if state.latest_ts is None or state.latest_ts == state.last_scored:
                    continue
                frame = state.frame()
                state.last_scored = state.latest_ts
                state.health = state.health_check(
                    self.cfg.system_frames_ratio, self.cfg.system_confidence_ratio
                )
            batch.append((state, frame))
        if batch:
            # Before scoring, so a system-wide degradation suppresses this check's alerts.
            try:
                self._check_system(states, anomaly)
            except Exception:
                log.exception("system health check failed")
        for state, frame in batch:
            try:
                self._score_camera(state, frame, congestion, anomaly)
            except Exception:
                SERVICE_ERRORS.labels("scoring").inc()
                log.exception("camera %s: scoring failed", state.camera_id)
        ANOMALIES_ACTIVE.labels("camera").set(sum(s.active is not None for s in states))
        ANOMALIES_ACTIVE.labels("system").set(int(self.system is not None))
        CAMERAS_WARMING.set(len(self.warming_up(states)))

    def warming_up(self, states: list[CameraState]) -> list[str]:
        """Cameras of the cameras table that can't raise traffic alerts yet: still in their
        warm-up, or without recent metrics (not started, or the feed paused). Cameras that only
        exist in a test's database are left out."""
        now = datetime.now(UTC)
        fresh_s = 2 * self.cfg.max_metric_age_seconds
        by_id = {s.camera_id: s for s in states}
        out = []
        for camera_id in self.settings:
            s = by_id.get(camera_id)
            recent = (
                s is not None
                and s.latest_ts is not None
                and (now - s.latest_ts).total_seconds() <= fresh_s
            )
            if s is None or not s.warm or not recent:
                out.append(camera_id)
        return sorted(out)

    # ------------------------------------------------------------------ system degradation

    def _check_system(self, states: list[CameraState], anomaly: LoadedModel) -> None:
        """Count cameras with a frame or confidence drop in this check; start, extend or end
        the system-wide degradation episode."""
        healths = [s.health for s in states if s.health is not None]
        if not healths:
            return
        ts = max(h.ts for h in healths)
        fresh = [
            s
            for s in states
            if s.health is not None and (ts - s.health.ts).total_seconds() <= SYSTEM_FRESH_S
        ]
        degraded = {s.camera_id: s.health for s in fresh if s.health.dropped}
        ep = self.system
        if len(degraded) >= self.cfg.system_min_cameras:
            severity = "high" if len(degraded) == len(fresh) else "medium"
            drops = [col for h in degraded.values() for col in h.dropped]
            if ep is None:
                lookback = ts - timedelta(seconds=self.cfg.system_lookback_seconds)
                self.system = ep = SystemEpisode(
                    anomaly_id=uuid.uuid4(),
                    started_at=ts,
                    cameras=set(degraded),
                    ratios={c: h.ratios for c, h in degraded.items()},
                    metric_name=max(sorted(set(drops)), key=drops.count),
                    severity=severity,
                    peak_cameras=len(degraded),
                    # Traffic alerts that started just before it was detected are likely the
                    # same glitch: they're closed when it clears.
                    linked={
                        s.camera_id: s.active.anomaly_id
                        for s in states
                        if s.active is not None and s.active.started_at >= lookback
                    },
                )
                self._publish_system(AlarmEvent.START, ts)
                return
            ep.clear_checks = 0
            new_cameras = set(degraded) - ep.cameras
            ep.cameras |= set(degraded)
            ep.ratios.update({c: h.ratios for c, h in degraded.items()})
            ep.peak_cameras = max(ep.peak_cameras, len(degraded))
            escalated = SEVERITY_ORDER.index(severity) > SEVERITY_ORDER.index(ep.severity)
            if escalated:
                ep.severity = severity
            due = ep.last_published is None or (ts - ep.last_published).total_seconds() >= (
                self.cfg.update_min_interval_seconds
            )
            if (new_cameras or escalated) and due:
                ep.updates += 1
                self._publish_system(AlarmEvent.UPDATE, ts)
        elif ep is not None:
            ep.clear_checks += 1
            if ep.clear_checks >= self.cfg.clear_after:
                self._end_system(ts, anomaly)

    def _end_system(self, ts: datetime, anomaly: LoadedModel) -> None:
        """End the system episode, closing the traffic alerts linked to it."""
        ep = self.system
        assert ep is not None
        closed = []
        for camera_id, anomaly_id in ep.linked.items():
            state = self.cameras.get(camera_id)
            a = state.active if state is not None else None
            if a is None or a.anomaly_id != anomaly_id:
                continue  # already ended on its own
            a.hold_reasons = []
            self._publish(
                state,
                AlarmEvent.END,
                state.tracker.peak_score,
                a.threshold,
                anomaly,
                a.versions,
                ts,
                extra={"closed_by": "system_degradation", "system_anomaly_id": str(ep.anomaly_id)},
            )
            # No cooldown: if the camera still scores high, that is a real alert again.
            state.tracker.close(ts.timestamp(), cooldown=False)
            state.active, state.frozen_baseline = None, None
            closed.append(str(anomaly_id))
        self._publish_system(AlarmEvent.END, ts, closed_alerts=closed)
        self.system = None

    def _publish_system(self, event: AlarmEvent, ts: datetime, **extra) -> None:
        ep = self.system
        assert ep is not None
        ep.last_published = ts
        window_start = ep.started_at - timedelta(seconds=HEALTH_RECENT_WINDOWS * WINDOW_S)
        observed = [r.get(ep.metric_name) for r in ep.ratios.values()]
        observed = [v for v in observed if v is not None]
        details = {
            "scope": "system",
            "health_ratios": ep.ratios,
            "limits": {
                "frames": self.cfg.system_frames_ratio,
                "mean_confidence": self.cfg.system_confidence_ratio,
                "min_cameras": self.cfg.system_min_cameras,
            },
            "linked_alerts": {c: str(a) for c, a in ep.linked.items()},
        }
        if event is AlarmEvent.END:
            details.update(suppressed_checks=ep.suppressed_checks, updates=ep.updates, **extra)
        message = Anomaly(
            anomaly_id=ep.anomaly_id,
            scope="system",
            camera_id=None,
            affected_cameras=sorted(ep.cameras),
            event=event.value,
            severity=ep.severity,
            anomaly_type="camera_degraded",
            hints=["camera_degraded"],
            metric_name=ep.metric_name,
            observed_value=min(observed) if observed else 0.0,
            expected_value=1.0,
            score=float(ep.peak_cameras),
            threshold=float(self.cfg.system_min_cameras),
            contributing_features={c: round(1 / len(ep.cameras), 4) for c in sorted(ep.cameras)},
            model_name=SYSTEM_MODEL[0],
            model_version=SYSTEM_MODEL[1],
            model_versions={SYSTEM_MODEL[0]: SYSTEM_MODEL[1]},
            started_at=ep.started_at,
            ended_at=ts if event is AlarmEvent.END else None,
            window_start=window_start,
            window_end=ts if ts > window_start else window_start + timedelta(seconds=1),
            timestamp=ts,
            details=details,
        )
        self.send(Topic.ANOMALIES, message)
        self._count(event, "system")
        log.warning(
            "ANOMALY %-6s scope=system type=camera_degraded severity=%s cameras=%s metric=%s "
            "linked=%s suppressed_checks=%d age=%.0fs id=%s",
            event.value,
            ep.severity,
            sorted(ep.cameras),
            ep.metric_name,
            sorted(ep.linked) or "-",
            ep.suppressed_checks,
            (ts - ep.started_at).total_seconds(),
            ep.anomaly_id,
        )
        self._store_anomaly(message)

    def _count(self, event: AlarmEvent, scope: str = "camera") -> None:
        ANOMALY_EVENTS.labels(scope, event.value).inc()
        if event is AlarmEvent.START:
            self.stats.started += 1
        elif event is AlarmEvent.END:
            self.stats.ended += 1
        else:
            self.stats.updated += 1

    def _score_camera(
        self, state: CameraState, frame, congestion: LoadedModel, anomaly: LoadedModel
    ):
        ts = frame["ts"].iloc[-1]
        with timed(INFERENCE_SECONDS, model=congestion.name):
            level_row = congestion.predictor.predict(frame).iloc[-1]
        level = str(level_row["level"])
        confidence = _float(level_row.get("confidence"))
        ts0 = frame["ts"].iloc[-1]
        if (
            state.active is not None
            and state.frozen_baseline is not None
            and (ts0 - state.active.started_at).total_seconds() > self.cfg.max_hold_seconds
        ):
            # A change that lasts this long is the new normal: let the rolling baseline take
            # it in, so the score can come down and the alert end (it can't otherwise).
            log.warning(
                "camera %s: alert %s open for over %.0fs; releasing its frozen baseline",
                state.camera_id,
                state.active.anomaly_id,
                self.cfg.max_hold_seconds,
            )
            state.frozen_baseline = None
            state.active.baseline_released = True
        ready = self._warmed_up(state, frame, anomaly, ts)
        with timed(INFERENCE_SECONDS, model=anomaly.name):
            result = anomaly.predictor.predict(state.model_input(frame)).iloc[-1].copy()
        self.stats.scored += 1
        valid = bool(result.get("valid", True))
        score, threshold = _float(result["score"]), _float(result["threshold"])
        if valid and score is not None and threshold is not None:
            score = self._vet_health(state, result, score, threshold)
        versions = {congestion.name: congestion.version, anomaly.name: anomaly.version}
        state.congestion_level = level
        self._upsert_congestion(
            state.camera_id, level, confidence, score if valid else None, threshold, versions, ts
        )
        if not valid or score is None or threshold is None:
            return
        if not ready:
            # The baselines aren't full yet: the camera is compared with a few minutes at best.
            self.stats.warming += 1
            state.tracker.suppress()
            return
        if self.system is not None and state.active is None:
            # A pipeline-wide degradation explains high scores now: no new traffic alerts.
            if score >= threshold:
                self.system.suppressed_checks += 1
            state.tracker.suppress()
            return
        hold_reasons = self._hold_reasons(state, frame, ts)
        event = state.tracker.update(ts.timestamp(), score, threshold, hold=bool(hold_reasons))
        if event is AlarmEvent.START:
            started = datetime.fromtimestamp(state.tracker.started_at, UTC)
            state.active = ActiveAnomaly(
                anomaly_id=uuid.uuid4(),
                started_at=started,
                anomaly_type="unknown",
                hints=[],
                metric_name="unknown",
                observed=0.0,
                expected=None,
                contributions={},
                congestion_level=level,
            )
            state.frozen_baseline = current_baselines(frame)
            state.active.versions = versions
            self._refresh_active(state.active, result, score, threshold)
            self._publish(state, AlarmEvent.START, score, threshold, anomaly, versions, ts)
        elif event is AlarmEvent.END and state.active is not None:
            state.active.hold_reasons = hold_reasons
            self._publish(
                state, AlarmEvent.END, state.tracker.peak_score, threshold, anomaly, versions, ts
            )
            state.active, state.frozen_baseline = None, None
        elif state.active is not None:
            a = state.active
            a.hold_reasons = hold_reasons
            before_rank, before_hints = SEVERITY_ORDER.index(a.severity), set(a.seen_hints)
            self._refresh_active(a, result, state.tracker.peak_score, threshold)
            if self._recovered(state, frame):
                a.recovered_since = a.recovered_since or ts
            else:
                a.recovered_since = None
            if (
                a.recovered_since is not None
                and (ts - a.recovered_since).total_seconds() >= self.cfg.recovery_seconds
            ):
                # Traffic is back to normal even if the score hasn't cleared (a frozen baseline
                # can keep it up): end the alert, with the usual cooldown.
                a.hold_reasons = []
                self.stats.recovered += 1
                self._publish(
                    state,
                    AlarmEvent.END,
                    state.tracker.peak_score,
                    threshold,
                    anomaly,
                    versions,
                    ts,
                    extra={"closed_by": "recovered"},
                )
                state.tracker.close(ts.timestamp())
                state.active, state.frozen_baseline = None, None
                return
            escalated = SEVERITY_ORDER.index(a.severity) > before_rank
            new_hints = set(a.seen_hints) - before_hints
            recovery_changed = a.recovered_since != a.published_recovered
            due = a.last_published is None or (ts - a.last_published).total_seconds() >= (
                self.cfg.update_min_interval_seconds
            )
            if (escalated or new_hints or recovery_changed) and due:
                a.updates += 1
                self._publish(
                    state,
                    AlarmEvent.UPDATE,
                    state.tracker.peak_score,
                    threshold,
                    anomaly,
                    versions,
                    ts,
                )

    def _warmed_up(self, state: CameraState, frame, anomaly: LoadedModel, ts) -> bool:
        """Whether the camera may raise traffic alerts: it has had ``warmup_seconds`` of
        metrics and is calibrated for the current model and its settings. Calibrates at the
        end of the warm-up, and again after a model reload or a settings change."""
        if not state.warm:
            start = state.warmup_from or ts
            if (ts - start).total_seconds() < self.cfg.warmup_seconds:
                return False
        key = f"{anomaly.label}|{state.settings}"
        if state.calibrated_for != key:
            if not self._calibrate(state, frame, anomaly):
                return False  # too little history to calibrate yet: keep warming up
            state.calibrated_for = key
        if not state.warm:
            state.warm = True
            calibration = state.calibration.describe() if state.calibration else "no calibration"
            log.info(
                "camera %s: warm-up complete after %.0fs; %s",
                state.camera_id,
                (ts - (state.warmup_from or ts)).total_seconds(),
                calibration,
            )
        return True

    def _calibrate(self, state: CameraState, frame, anomaly: LoadedModel) -> bool:
        py = python_model(anomaly.predictor)
        if py is None:  # the rule-based fallback has nothing to calibrate
            state.calibration = None
            return True
        pristine = self._pristine.get(anomaly.label)
        if pristine is None:  # first calibration with this model: keep its own calibration
            pristine = self._pristine[anomaly.label] = copy.deepcopy(py.calibration)
        result = calibrate_camera(
            py,
            pristine,
            state.camera_id,
            frame,
            state.settings,
            min_sequences=self.cfg.calibration_min_sequences,
        )
        if result is None:
            return False
        if state.warm:  # a recalibration: the model was reloaded or the settings changed
            log.info(
                "camera %s: recalibrated for %s: %s",
                state.camera_id,
                anomaly.label,
                result.describe(),
            )
        state.calibration = result
        return True

    def _recovered(self, state: CameraState, frame) -> bool:
        """Whether an active traffic alert's metrics are back to normal: loop and camera speed
        (the last 3 windows) at least ``recovery_ratio`` x the frozen baseline, and stopped
        vehicles and pedestrians within ``hold_stopped_margin`` of it. Signals the camera
        ignores, or without data, aren't judged; at least one speed must be."""
        a, frozen = state.active, state.frozen_baseline
        if a is None or frozen is None or not (a.seen_hints & TRAFFIC_HINTS):
            return False
        recent = frame.tail(3)
        judged = 0
        for col, signal in (("loop_speed_kmh", None), ("avg_speed_kmh", "camera_speed")):
            if signal and state.settings.ignores(signal):
                continue
            now, base = recent[col].astype(float).mean(), frozen.get(col)
            if not base or pd.isna(now):
                continue
            judged += 1
            if now < self.cfg.recovery_ratio * base:
                return False
        for col, signal in (
            ("stopped_vehicles", "stopped_vehicles"),
            ("pedestrians_in_roadway", "pedestrians"),
        ):
            now, base = recent[col].astype(float).mean(), frozen.get(col)
            if state.settings.ignores(signal) or base is None or pd.isna(now):
                continue
            if now > base + self.cfg.hold_stopped_margin:
                return False
        return judged > 0

    def _vet_health(self, state: CameraState, result, score: float, threshold: float) -> float:
        """Hold the model's camera_degraded explanation to the health check (frames or
        detection confidence well below the camera's baseline, the same test as the
        system-wide check). Without a real drop the hint is removed, and a score that only
        crosses the threshold through pipeline features is pipeline noise: it's capped just
        below the threshold so it neither starts nor holds an alert."""
        if state.health is not None and state.health.dropped:
            return score
        hints = json.loads(result.get("hints") or "[]")
        if "camera_degraded" in hints:
            hints = [h for h in hints if h != "camera_degraded"]
            result["hints"] = json.dumps(hints)
        top = str(result.get("top_feature") or "")
        if score >= threshold and not hints and top in HEALTH_FEATURES:
            self.stats.health_noise += 1
            log.debug(
                "camera %s: score %.2f >= %.2f only through %s with a healthy pipeline; ignored",
                state.camera_id,
                score,
                threshold,
                top,
            )
            return min(score, threshold * 0.999)
        return score

    def _refresh_active(self, a: ActiveAnomaly, result, peak: float, threshold: float) -> None:
        """Take the model's latest explanation; severity only ever rises within an episode."""
        a.threshold = threshold
        hints = json.loads(result.get("hints") or "[]")
        if hints:
            a.hints, a.anomaly_type = hints, hints[0]
            a.seen_hints.update(hints)
        contributions = json.loads(result.get("contributions") or "{}")
        if contributions:
            a.contributions = contributions
            a.metric_name = str(result.get("top_feature") or a.metric_name)
            a.observed = _float(result.get("observed")) or 0.0
            a.expected = _float(result.get("expected"))
        a.fired_by = result.get("fired_by") or a.fired_by
        a.score_parts = {
            k: round(v, 4)
            for k in ("reconstruction_score", "level_score")
            if (v := _float(result.get(k))) is not None
        }
        severity = self.severity(peak / threshold if threshold else 0.0)
        if SEVERITY_ORDER.index(severity) > SEVERITY_ORDER.index(a.severity):
            a.severity = severity

    def severity(self, ratio: float) -> str:
        low, medium, high = self.cfg.severity_limits
        if ratio >= high:
            return "critical"
        if ratio >= medium:
            return "high"
        if ratio >= low:
            return "medium"
        return "low"

    def _hold_reasons(self, state: CameraState, frame, ts) -> list[str]:
        """Why an active alert must stay open, judged against the baseline frozen at its start."""
        frozen = state.frozen_baseline
        if state.active is None or frozen is None:
            return []
        if (ts - state.active.started_at).total_seconds() > self.cfg.max_hold_seconds:
            return []
        recent = frame.tail(3)
        reasons = []
        for col, name in (("loop_speed_kmh", "loop_speed"), ("avg_speed_kmh", "camera_speed")):
            if state.settings.ignores(name):
                continue
            now, base = recent[col].astype(float).mean(), frozen.get(col)
            if base and pd.notna(now) and now < self.cfg.hold_speed_ratio * base:
                reasons.append(f"{name}_below_{self.cfg.hold_speed_ratio:g}x_baseline")
        stopped, base = (
            recent["stopped_vehicles"].astype(float).mean(),
            frozen.get("stopped_vehicles"),
        )
        if (
            not state.settings.ignores("stopped_vehicles")
            and base is not None
            and pd.notna(stopped)
            and stopped > base + self.cfg.hold_stopped_margin
        ):
            reasons.append("stopped_vehicles_above_baseline")
        return reasons

    def _publish(
        self, state, event, score, threshold, model: LoadedModel, versions, ts, extra=None
    ) -> None:
        a = state.active
        assert a is not None
        a.last_published = ts
        if event is AlarmEvent.START:
            window_start = ts - timedelta(seconds=SEQUENCE_LENGTH * WINDOW_S)
        else:
            window_start = a.started_at
        details = {
            "model_source": model.source,
            "fired_by": a.fired_by,
            "score_parts": a.score_parts,
        }
        if a.hold_reasons:
            details["hold"] = a.hold_reasons
        if event is not AlarmEvent.START:
            recovered = a.recovered_since
            details["recovered_since"] = recovered.isoformat() if recovered else None
            a.published_recovered = recovered
        if event is not AlarmEvent.UPDATE:
            details["frozen_baseline"] = {
                k: round(v, 3) for k, v in (state.frozen_baseline or {}).items() if v is not None
            }
        if event is AlarmEvent.END:
            details["held_checks"] = state.tracker.held_checks
            details["updates"] = a.updates
            details["baseline_released"] = a.baseline_released
        details.update(extra or {})
        message = Anomaly(
            anomaly_id=a.anomaly_id,
            camera_id=state.camera_id,
            event=event.value,
            severity=a.severity,
            anomaly_type=a.anomaly_type,
            hints=a.hints,
            metric_name=a.metric_name,
            observed_value=a.observed,
            expected_value=a.expected,
            score=round(score, 4),
            threshold=round(threshold, 4),
            contributing_features=a.contributions,
            congestion_level=state.congestion_level
            if event is not AlarmEvent.START
            else a.congestion_level,
            model_name=model.name,
            model_version=model.version,
            model_versions=versions,
            started_at=a.started_at,
            ended_at=ts if event is AlarmEvent.END else None,
            window_start=window_start,
            window_end=ts if ts > window_start else window_start + timedelta(seconds=1),
            timestamp=ts,
            details=details,
        )
        self.send(Topic.ANOMALIES, message)
        self._count(event)
        log.warning(
            "ANOMALY %-6s camera=%s type=%s severity=%s hints=%s score=%.3f threshold=%.3f "
            "fired_by=%s hold=%s age=%.0fs id=%s",
            event.value,
            state.camera_id,
            a.anomaly_type,
            a.severity,
            a.hints,
            score,
            threshold,
            a.fired_by,
            a.hold_reasons or "-",
            (ts - a.started_at).total_seconds(),
            a.anomaly_id,
        )
        self._store_anomaly(message)

    # ------------------------------------------------------------------ persistence

    def _store_anomaly(self, m: Anomaly) -> None:
        try:
            with self.engine.begin() as conn:
                if m.event == "start":
                    conn.execute(
                        insert(AnomalyRow).values(
                            id=m.anomaly_id,
                            camera_id=m.camera_id,
                            scope=m.scope,
                            affected_cameras=m.affected_cameras,
                            anomaly_type=m.anomaly_type,
                            metric_name=m.metric_name,
                            observed_value=m.observed_value,
                            expected_value=m.expected_value,
                            score=m.score,
                            threshold=m.threshold,
                            window_start=m.window_start,
                            window_end=m.window_end,
                            detected_at=datetime.now(UTC),
                            model_name=m.model_name,
                            model_version=m.model_version,
                            details=m.details,
                            status="active",
                            hints=[h.value for h in m.hints],
                            contributing_features=m.contributing_features,
                            congestion_level=m.congestion_level,
                            model_versions=m.model_versions,
                            severity=m.severity,
                        )
                    )
                else:
                    values = {
                        "score": m.score,
                        "window_end": m.window_end,
                        "severity": m.severity,
                        "anomaly_type": m.anomaly_type,
                        "hints": [h.value for h in m.hints],
                        "contributing_features": m.contributing_features,
                        "affected_cameras": m.affected_cameras,
                        "details": AnomalyRow.details.op("||")(m.details),
                    }
                    if m.event == "update":
                        values["update_count"] = AnomalyRow.update_count + 1
                    else:
                        values.update(status="ended", ended_at=m.ended_at)
                    conn.execute(
                        update(AnomalyRow).where(AnomalyRow.id == m.anomaly_id).values(**values)
                    )
        except IntegrityError:
            self._warn_once(m.camera_id, "anomaly for a camera not in the cameras table")
        except Exception as exc:
            self.stats.db_errors += 1
            SERVICE_ERRORS.labels("database").inc()
            log.error("could not store anomaly %s: %s", m.anomaly_id, exc)

    def _upsert_congestion(self, camera_id, level, confidence, score, threshold, versions, ts):
        stmt = insert(CameraCongestion).values(
            camera_id=camera_id,
            level=level,
            confidence=confidence,
            anomaly_score=score,
            anomaly_threshold=threshold,
            model_versions=versions,
            window_end=ts,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[CameraCongestion.camera_id],
            set_={
                c: stmt.excluded[c]
                for c in (
                    "level",
                    "confidence",
                    "anomaly_score",
                    "anomaly_threshold",
                    "model_versions",
                    "window_end",
                )
            }
            | {"updated_at": datetime.now(UTC)},
        )
        try:
            with self.engine.begin() as conn:
                conn.execute(stmt)
                conn.execute(
                    insert(CongestionHistory).values(
                        camera_id=camera_id,
                        window_end=ts,
                        level=level,
                        confidence=confidence,
                        anomaly_score=score,
                        anomaly_threshold=threshold,
                    )
                )
        except IntegrityError:
            self._warn_once(camera_id, "congestion for a camera not in the cameras table")
        except Exception as exc:
            self.stats.db_errors += 1
            SERVICE_ERRORS.labels("database").inc()
            log.error("could not store congestion for %s: %s", camera_id, exc)

    def _warn_once(self, camera_id: str | None, what: str) -> None:
        if camera_id not in self._db_warned:
            self._db_warned.add(camera_id or "")
            log.warning("camera %s: %s, not stored", camera_id, what)

    def close_orphaned_anomalies(self) -> int:
        """Mark anomalies left 'active' by a previous run as ended (their state was lost on
        restart) and publish an end event for each, so downstream services don't keep them
        open forever. Returns how many were closed."""
        now = datetime.now(UTC)
        try:
            with self.engine.begin() as conn:
                rows = conn.execute(
                    update(AnomalyRow)
                    .where(AnomalyRow.status == "active")
                    .values(
                        status="ended",
                        ended_at=now,
                        details=AnomalyRow.details.op("||")({"closed_by": "service restart"}),
                    )
                    .returning(*AnomalyRow.__table__.c)
                ).all()
        except Exception as exc:
            log.error("could not close orphaned anomalies: %s", exc)
            return 0
        for row in rows:
            try:
                message = Anomaly(
                    anomaly_id=row.id,
                    camera_id=row.camera_id,
                    event="end",
                    scope=row.scope,
                    affected_cameras=row.affected_cameras,
                    severity=row.severity,
                    anomaly_type=row.anomaly_type,
                    hints=row.hints,
                    metric_name=row.metric_name,
                    observed_value=row.observed_value,
                    expected_value=row.expected_value,
                    score=row.score,
                    threshold=row.threshold,
                    contributing_features=row.contributing_features,
                    congestion_level=row.congestion_level,
                    model_name=row.model_name,
                    model_version=row.model_version,
                    model_versions=row.model_versions,
                    # a start event's window_start is SEQUENCE_LENGTH windows before it
                    started_at=row.window_start + timedelta(seconds=SEQUENCE_LENGTH * WINDOW_S),
                    ended_at=now,
                    window_start=row.window_start,
                    window_end=max(now, row.window_start + timedelta(seconds=1)),
                    timestamp=now,
                    details=row.details,
                )
                self.send(Topic.ANOMALIES, message)
            except Exception as exc:  # an old row that no longer fits the schema
                log.error("could not publish end for orphaned anomaly %s: %s", row.id, exc)
        if rows:
            log.info("closed %d anomaly(ies) left active by a previous run", len(rows))
        return len(rows)

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
                metrics={
                    "role": model.role,
                    "loaded_by": "anomaly_service",
                    **{k: v for k, v in model.metadata.items() if isinstance(v, int | float | str)},
                },
            )
        except Exception as exc:
            log.warning("could not record %s in model_versions: %s", model.label, exc)

    def log_stats(self) -> None:
        with self._lock:
            levels = {
                cid: s.congestion_level
                for cid, s in sorted(self.cameras.items())
                if s.congestion_level
            }
            active = [cid for cid, s in self.cameras.items() if s.active]
            warming = self.warming_up(list(self.cameras.values()))
        log.info(
            "stats %s congestion=%s active_anomalies=%s warming_up=%s system=%s models=%s",
            self.stats.reset(),
            levels,
            active or "none",
            warming or "none",
            sorted(self.system.cameras) if self.system else "ok",
            {r: self.models.get(r).label for r in ("congestion", "anomaly")},
        )


def _float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f
