"""Correlation and lifecycle, with no I/O: messages and clock ticks in, Changes out.

A camera can have several open incidents (an unrelated burst of reports doesn't get lost
behind an open incident of another type). New evidence joins the open incident on its camera
it fits best; evidence pools hold what hasn't become an incident yet (anomalies, reports,
recent metrics); a sliding window (``window_seconds``) bounds what can be grouped together.

- Created (suspected) when a modality fires: a traffic anomaly is active on a camera with
  no open incident (not during a system-wide degradation on that camera), or a burst of
  same-type reports arrives (``min_text_reports`` within ``text_burst_seconds`` from
  ``min_text_sources`` sources) that fits none of the camera's open incidents.
- Confirmed when both modalities are present, each reasonably strong, and agree
  (``confirm_min_agreement``), or when one modality alone scores ``strong_single``.
- Resolved when no attached anomaly is active, camera metrics are back near the baseline
  from before the incident, and either a clearing report has arrived (then after only
  ``clear_min_seconds``) or nothing new has arrived for ``quiet_seconds``. With a clearing
  report, an anomaly whose traffic metrics have recovered (anomaly_service's
  ``recovered_since``) for ``clear_min_seconds`` no longer holds the incident open.
- A camera's anomalies alone create no incident until it has ``vision_warmup_seconds`` of
  uninterrupted metrics.
- Reports with no camera match wait in an "unlocated" pool and join an open incident whose
  reports name the same place.
- Start time: the first evidence that supports the incident's (current, finally its last) type,
  not the earliest evidence attached; that is kept as ``first_evidence_at``. Attaching takes in
  the camera's recent reports, so the earliest can be a background report from minutes before.
"""

import logging
import statistics
import uuid
from collections import deque
from datetime import datetime, timedelta

from rapidfuzz import fuzz

from fusion_service import confidence
from fusion_service.config import FusionConfig
from fusion_service.model import AnomalyEvidence, Change, FusedIncident, MetricPoint, ReportEvidence
from fusion_service.types import (
    hint_agreement,
    max_severity,
    robust_text_severity,
    type_compatibility,
    type_from_hints,
    vision_severity,
)
from ml.nlp.matching import normalize
from traffic_common.schemas import Anomaly, NLPResult, TrafficMetrics

log = logging.getLogger(__name__)
GENERIC_PLACES = {"london", "uk", "england", "traffic", "londontraffic", "the city"}


def places_from_entities(entities: list) -> list[str]:
    """Normalised LOCATION entity texts, without generic ones ("London") that match anything."""
    out = set()
    for e in entities:
        label, value = (e.label, e.text) if hasattr(e, "label") else (e["label"], e["text"])
        if label == "LOCATION":
            norm = normalize(value)
            if len(norm) >= 4 and norm not in GENERIC_PLACES:
                out.add(norm)
    return sorted(out)


def places_of(result: NLPResult) -> list[str]:
    return places_from_entities(result.entities)


class FusionEngine:
    def __init__(self, cfg: FusionConfig, camera_names: dict[str, str] | None = None) -> None:
        self.cfg = cfg
        self.camera_names = camera_names or {}
        self.incidents: dict[str, FusedIncident] = {}  # open incidents by incident id
        self.anomalies: dict[str, AnomalyEvidence] = {}  # camera-scoped, recent
        self.reports: dict[str, dict[str, ReportEvidence]] = {}  # camera -> pool
        self.unlocated: dict[str, ReportEvidence] = {}
        self.metrics: dict[str, deque[MetricPoint]] = {}
        self.congestion: dict[str, str] = {}
        self.system: dict | None = None  # active system-wide degradation
        self._changes: list[Change] = []
        self._pending: dict[str, list[dict]] = {}  # incident id -> new evidence rows
        self._reasons: dict[str, list[str]] = {}
        self.ignored = 0  # evidence for cameras not in the cameras table

    def known(self, camera_id: str) -> bool:
        if camera_id in self.camera_names:
            return True
        self.ignored += 1
        return False

    def open_on(self, camera_id: str) -> list[FusedIncident]:
        """The camera's open incidents, oldest first."""
        return sorted(
            (i for i in self.incidents.values() if i.camera_id == camera_id),
            key=lambda i: i.suspected_at,
        )

    @property
    def window(self) -> timedelta:
        return timedelta(seconds=self.cfg.window_seconds)

    def drain(self) -> list[Change]:
        out, self._changes = self._changes, []
        return out

    # ------------------------------------------------------------------ inputs

    def on_anomaly(self, m: Anomaly, now: datetime) -> None:
        if m.scope == "system":
            self._on_system(m, now)
            return
        if m.camera_id is None or not self.known(m.camera_id):
            return
        key = str(m.anomaly_id)
        a = self.anomalies.get(key)
        ratio = m.score / m.threshold if m.threshold else 0.0
        if a is None:
            a = AnomalyEvidence(
                anomaly_id=key,
                camera_id=m.camera_id,
                started_at=m.started_at or m.window_start,
                last_seen=m.timestamp,
                peak_ratio=ratio,
                hints=[str(h) for h in m.hints],
                severity=m.severity,
                congestion=m.congestion_level,
                degraded=self._degraded(m.camera_id) or "camera_degraded" in m.hints,
            )
            self.anomalies[key] = a
        a.last_seen = max(a.last_seen, m.timestamp)
        a.peak_ratio = max(a.peak_ratio, ratio)
        a.hints = sorted(set(a.hints) | {str(h) for h in m.hints})
        a.severity = max_severity(a.severity, m.severity) if m.severity else a.severity
        a.congestion = m.congestion_level or a.congestion
        # A camera reporting itself as degraded makes its other hints unreliable too.
        a.degraded = a.degraded or self._degraded(m.camera_id) or "camera_degraded" in a.hints
        if m.event == "end":
            a.active, a.ended_at = False, m.ended_at or m.timestamp
        elif "recovered_since" in m.details:  # updates say whether traffic has recovered
            since = m.details["recovered_since"]
            a.recovered_since = datetime.fromisoformat(since) if since else None
        inc = self._incident_for_anomaly(a)
        if inc is not None and a.traffic:
            self._attach_anomaly(inc, a, "anomaly " + m.event)
        self._update(m.camera_id, now)

    def on_nlp(self, r: NLPResult, now: datetime) -> None:
        if r.incident_type == "none" or r.incident_confidence < self.cfg.min_report_confidence:
            return
        ev = ReportEvidence(
            report_id=str(r.report_id),
            camera_id=r.camera_id,
            ts=r.timestamp,
            incident_type=str(r.incident_type),
            confidence=r.incident_confidence,
            severity=r.severity,
            source=r.source,
            camera_confidence=r.camera_confidence or 0.0,
            stage=r.stage,
            places=places_of(r),
        )
        if r.camera_id is None:
            if ev.stage == "active" and ev.places:
                self.unlocated[ev.report_id] = ev
                for inc in list(self.incidents.values()):
                    self._attach_unlocated(inc, now)
            return
        if not self.known(r.camera_id):
            return
        self.reports.setdefault(r.camera_id, {})[ev.report_id] = ev
        self._route_report(ev, now)
        self._update(r.camera_id, now)

    def on_metrics(self, m: TrafficMetrics) -> None:
        q = self.metrics.setdefault(m.camera_id, deque(maxlen=400))
        q.append(MetricPoint(m.window_end, m.avg_speed_kmh, m.stopped_vehicles, str(m.message_id)))

    def set_congestion(self, camera_id: str, level: str | None) -> None:
        if level:
            self.congestion[camera_id] = level

    def tick(self, now: datetime) -> None:
        """Time-driven transitions (quiet periods) and pool expiry."""
        cameras = (
            {i.camera_id for i in self.incidents.values()}
            | set(self.reports)
            | {a.camera_id for a in self.anomalies.values()}
        )
        for camera_id in cameras:
            self._update(camera_id, now)
        self._expire(now)

    # ------------------------------------------------------------------ system degradation

    def _degraded(self, camera_id: str) -> bool:
        return self.system is not None and camera_id in self.system["cameras"]

    def _on_system(self, m: Anomaly, now: datetime) -> None:
        if m.event == "end":
            if self.system and self.system["anomaly_id"] == str(m.anomaly_id):
                log.info("system degradation %s ended", m.anomaly_id)
                self.system = None
            return
        self.system = {
            "anomaly_id": str(m.anomaly_id),
            "cameras": set(m.affected_cameras),
            "started_at": m.started_at or m.timestamp,
        }
        for camera_id in m.affected_cameras:
            marked_any = False
            for inc in self.open_on(camera_id):
                marked = False
                for a in inc.anomalies.values():
                    if a.active and not a.degraded:
                        a.degraded = marked = True
                if marked:
                    marked_any = True
                    self._note(inc, "vision degraded (system-wide degradation)")
            if marked_any:
                self._update(camera_id, now)

    # ------------------------------------------------------------------ attaching evidence

    def _note(self, inc: FusedIncident, reason: str) -> None:
        reasons = self._reasons.setdefault(str(inc.incident_id), [])
        if reason not in reasons:
            reasons.append(reason)

    def _evidence(self, inc: FusedIncident, row: dict) -> None:
        self._pending.setdefault(str(inc.incident_id), []).append(row)

    @staticmethod
    def _saw(inc: FusedIncident, ts: datetime) -> None:
        first = inc.first_evidence_at
        inc.first_evidence_at = ts if first is None else min(first, ts)

    def _attach_anomaly(self, inc: FusedIncident, a: AnomalyEvidence, reason: str) -> None:
        new = a.anomaly_id not in inc.anomalies
        inc.anomalies[a.anomaly_id] = a
        inc.last_evidence_at = max(inc.last_evidence_at, a.last_seen)
        if new:
            self._saw(inc, a.started_at)
            self._evidence(
                inc,
                {
                    "evidence_type": "anomaly",
                    "anomaly_id": a.anomaly_id,
                    "weight": None,
                    "details": {"hints": a.hints, "degraded": a.degraded},
                },
            )
            self._note(inc, reason if reason != "anomaly start" else "anomaly attached")

    def _incident_for_anomaly(self, a: AnomalyEvidence) -> FusedIncident | None:
        """The open incident an anomaly belongs to: the one it's already attached to, else
        the camera's open incident its hints support best (the newest on a tie)."""
        opens = self.open_on(a.camera_id)
        for inc in opens:
            if a.anomaly_id in inc.anomalies:
                return inc
        if not opens:
            return None
        return max(reversed(opens), key=lambda i: hint_agreement(a.hints, i.incident_type))

    def _route_report(self, ev: ReportEvidence, now: datetime) -> FusedIncident | None:
        """Attach a located report to the camera's open incident it fits best."""
        opens = self.open_on(ev.camera_id or "")
        for inc in opens:
            if ev.report_id in inc.reports:
                return inc
        scored = sorted(
            (
                (type_compatibility(i.incident_type, ev.incident_type), n, i)
                for n, i in enumerate(opens)
            ),
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )
        for _, _, inc in scored:
            if self._try_attach_report(inc, ev, now):
                return inc
        return None

    def _report_fits(self, inc: FusedIncident, ev: ReportEvidence) -> bool:
        if ev.stage == "cleared":
            return True  # a clearing report for this camera is about this incident
        return type_compatibility(inc.incident_type, ev.incident_type) >= self.cfg.min_compatibility

    def _try_attach_report(self, inc: FusedIncident, ev: ReportEvidence, now: datetime) -> bool:
        earliest = inc.first_evidence_at or inc.suspected_at
        if ev.report_id in inc.reports or ev.ts < earliest - self.window:
            return False
        if not self._report_fits(inc, ev):
            return False
        inc.reports[ev.report_id] = ev
        self._saw(inc, ev.ts)
        if ev.stage == "cleared":
            inc.cleared_at = inc.cleared_at or ev.ts
            self._note(inc, "clearing report")
        else:
            inc.last_evidence_at = max(inc.last_evidence_at, ev.ts)
            self._note(inc, "report attached")
        self._evidence(
            inc,
            {
                "evidence_type": "report",
                "report_id": ev.report_id,
                "weight": round(ev.confidence * max(ev.camera_confidence, 0.0), 4),
                "details": {
                    "type": ev.incident_type,
                    "stage": ev.stage,
                    "source": ev.source,
                    "unlocated": ev.unlocated,
                },
            },
        )
        if ev.places and not ev.unlocated:
            self._attach_unlocated(inc, now)
        return True

    def _attach_unlocated(self, inc: FusedIncident, now: datetime) -> None:
        """Unlocated reports whose place entities match this incident's places."""
        places = inc.places()
        if not places:
            return
        for key, ev in list(self.unlocated.items()):
            best = max((fuzz.ratio(p, q) / 100 for p in ev.places for q in places), default=0)
            if best < self.cfg.unlocated_match_score:
                continue
            ev.camera_id, ev.unlocated = inc.camera_id, True
            ev.camera_confidence = round(best * self.cfg.unlocated_camera_confidence_factor, 4)
            if self._try_attach_report(inc, ev, now):
                del self.unlocated[key]
                self._note(inc, "unlocated report attached")

    def _attach_metric(self, inc: FusedIncident, moment: str) -> None:
        q = self.metrics.get(inc.camera_id)
        if not q:
            return
        point = q[-1]
        if point.message_id not in inc.metric_ids:
            inc.metric_ids.append(point.message_id)
            self._evidence(
                inc,
                {
                    "evidence_type": "metric",
                    "metric_message_id": point.message_id,
                    "weight": None,
                    "details": {"at": moment, "avg_speed_kmh": point.avg_speed},
                },
            )

    # ------------------------------------------------------------------ firing

    def _text_burst(self, camera_id: str, now: datetime) -> str | None:
        """The type of a burst of same-type reports on the camera that none of its open
        incidents has taken in (they would fit one of them otherwise), if there is one."""
        since = now - timedelta(seconds=self.cfg.text_burst_seconds)
        opens = self.open_on(camera_id)
        taken = {k for inc in opens for k in inc.reports}
        by_type: dict[str, list[ReportEvidence]] = {}
        for ev in self.reports.get(camera_id, {}).values():
            if ev.stage == "active" and ev.ts >= since and ev.report_id not in taken:
                by_type.setdefault(ev.incident_type, []).append(ev)
        best = None
        for t, evs in by_type.items():
            sources = {e.source for e in evs}
            if len(evs) >= self.cfg.min_text_reports and len(sources) >= self.cfg.min_text_sources:
                if best is None or len(evs) > len(by_type[best]):
                    best = t
        return best

    def _warmed_up(self, camera_id: str, now: datetime) -> bool:
        """The camera has had ``vision_warmup_seconds`` of metrics without a gap (0: off)."""
        if self.cfg.vision_warmup_seconds <= 0:
            return True
        q = self.metrics.get(camera_id)
        if not q or (now - q[-1].ts).total_seconds() > self.cfg.metrics_gap_seconds:
            return False
        start = q[-1].ts
        for p in reversed(q):
            if (start - p.ts).total_seconds() > self.cfg.metrics_gap_seconds:
                break
            start = p.ts
        return (now - start).total_seconds() >= self.cfg.vision_warmup_seconds

    def _vision_firing(self, camera_id: str, now: datetime) -> list[AnomalyEvidence]:
        if not self._warmed_up(camera_id, now):
            return []
        fresh = now - timedelta(seconds=self.cfg.max_message_age_seconds)
        return [
            a
            for a in self.anomalies.values()
            if a.camera_id == camera_id
            and a.active
            and a.traffic
            and not a.degraded
            and a.started_at >= fresh  # a long-running alert isn't a new incident
        ]

    # ------------------------------------------------------------------ the state machine

    def _update(self, camera_id: str, now: datetime) -> None:
        for inc in self.open_on(camera_id):
            self._step(inc, now)
        inc = self._maybe_create(camera_id, now)
        if inc is not None:
            self._step(inc, now)

    def _step(self, inc: FusedIncident, now: datetime) -> None:
        self._recompute(inc, now)
        self._lifecycle(inc, now)
        self._emit(inc)

    def _maybe_create(self, camera_id: str, now: datetime) -> FusedIncident | None:
        opens = self.open_on(camera_id)
        # A new anomaly on a camera with open incidents joins one of them instead.
        vision = [] if opens else self._vision_firing(camera_id, now)
        burst = self._text_burst(camera_id, now)
        if not vision and burst is None:
            return None
        if burst is not None:
            incident_type, why = burst, f"text burst ({burst})"
        else:
            hints = sorted({h for a in vision for h in a.hints})
            t = type_from_hints(hints)
            incident_type, why = (t.value if t else "congestion"), "traffic anomaly"
        starts = [a.started_at for a in vision]
        inc = FusedIncident(
            incident_id=uuid.uuid4(),
            camera_id=camera_id,
            status="suspected",
            incident_type=incident_type,
            severity="low",
            start_time=min(starts) if starts else now,
            suspected_at=now,
            last_evidence_at=now,
            baseline=self._baseline(camera_id, now),
        )
        self.incidents[str(inc.incident_id)] = inc
        self._note(inc, f"created: {why}")
        horizon = now - self.window
        taken_anomalies = {k for o in opens for k in o.anomalies}
        taken_reports = {k for o in opens for k in o.reports}
        for a in self.anomalies.values():
            if (
                a.camera_id == camera_id
                and a.traffic
                and a.anomaly_id not in taken_anomalies
                and (a.active or (a.ended_at or now) >= horizon)
            ):
                self._attach_anomaly(inc, a, "anomaly attached")
        for ev in sorted(self.reports.get(camera_id, {}).values(), key=lambda e: e.ts):
            if ev.ts >= horizon and ev.stage == "active" and ev.report_id not in taken_reports:
                self._try_attach_report(inc, ev, now)
        self._attach_unlocated(inc, now)
        self._attach_metric(inc, "created")
        log.info("incident %s created on %s (%s)", inc.incident_id, camera_id, why)
        return inc

    def _recompute(self, inc: FusedIncident, now: datetime) -> None:
        parts = confidence.breakdown(
            list(inc.anomalies.values()), list(inc.reports.values()), now, self.cfg
        )
        inc.components, inc.fused = parts, parts["fusion"]["fused"]
        text_type, vision_type = parts["text"].get("type"), parts["vision"].get("type")
        new_type = text_type or vision_type or inc.incident_type
        if new_type != inc.incident_type:
            self._note(inc, f"type {inc.incident_type} -> {new_type}")
            inc.incident_type = new_type
        inc.start_time = self._start_time(inc, now)
        active = [r for r in inc.reports.values() if r.stage == "active"]
        text_sev = robust_text_severity(
            [r.severity for r in active if r.severity], self.cfg.min_severity_reports
        )
        clean = [a for a in inc.anomalies.values() if a.traffic and not a.degraded]
        anomaly_sev = max_severity(*(a.severity for a in clean)) if clean else None
        congestion = self.congestion.get(inc.camera_id) or next(
            (a.congestion for a in clean if a.congestion), None
        )
        vis_sev = vision_severity(anomaly_sev, congestion) if clean else None
        severity = max_severity(inc.severity, text_sev, vis_sev)
        if severity != inc.severity:
            self._note(inc, f"severity {inc.severity} -> {severity}")
            inc.severity = severity
        inc.components["severity"] = {"text": text_sev, "vision": vis_sev, "congestion": congestion}
        recent = parts["text"].get("recent_supporting_reports", 0)
        if inc.active_anomaly or recent >= self.cfg.min_text_reports:
            inc.last_active_at = now

    def _start_time(self, inc: FusedIncident, now: datetime) -> datetime:
        """The first evidence supporting the incident's type: active reports of that type and
        reliable anomalies whose hints agree with it (``confirm_min_agreement``). Of those,
        only the run that led to the incident firing counts: walking back from the evidence
        around ``suspected_at``, a gap over ``start_gap_seconds`` ends it, so a same-type
        background report minutes earlier doesn't become the start. Without any supporting
        evidence (yet), the earliest evidence attached."""
        t = inc.incident_type
        times = sorted(
            [r.ts for r in inc.reports.values() if r.stage == "active" and r.incident_type == t]
            + [
                a.started_at
                for a in inc.anomalies.values()
                if a.traffic
                and not a.degraded
                and hint_agreement(a.hints, t) >= self.cfg.confirm_min_agreement
            ]
        )
        if not times:
            return min(inc.first_evidence_at or inc.suspected_at, now)
        gap = timedelta(seconds=self.cfg.start_gap_seconds)
        # The supporting evidence nearest the moment the incident fired (the burst, or the
        # anomaly), or the latest before it.
        i = next((k for k, ts in enumerate(times) if ts >= inc.suspected_at - gap), len(times) - 1)
        while i > 0 and times[i] - times[i - 1] <= gap:
            i -= 1
        return min(times[i], now)

    def _lifecycle(self, inc: FusedIncident, now: datetime) -> None:
        c = inc.components
        vision, text, agreement = c["vision"]["score"], c["text"]["score"], c["agreement"]["score"]
        if inc.status == "suspected":
            both = (
                vision >= self.cfg.vision_confirm_min
                and text >= self.cfg.text_confirm_min
                and agreement >= self.cfg.confirm_min_agreement
            )
            strong = max(vision, text) >= self.cfg.strong_single
            if both or strong:
                inc.status, inc.confirmed_at = "confirmed", now
                inc.at_confirmation = {
                    "at": now.isoformat(),
                    "fused": inc.fused,
                    "rule": "both modalities agree"
                    if both
                    else f"strong {c['fusion']['primary_modality']}",
                    "components": c,
                }
                self._note(inc, f"confirmed ({inc.at_confirmation['rule']})")
                self._attach_metric(inc, "confirmed")
        if inc.status in ("suspected", "confirmed"):
            reason = self._resolution(inc, now)
            if reason:
                inc.status, inc.end_time, inc.resolution = "resolved", now, reason
                self._note(inc, f"resolved ({reason})")
                self._attach_metric(inc, "resolved")

    def _resolution(self, inc: FusedIncident, now: datetime) -> str | None:
        active = inc.active_anomalies
        # An active anomaly holds the incident open unless a clearing report has arrived and
        # every active one reports its traffic recovered (it ends a little later).
        recovered = (
            inc.cleared_at is not None
            and bool(active)
            and all(a.recovered_since is not None for a in active)
        )
        if active and not recovered:
            return None
        near, detail = self._near_baseline(inc, now)
        if not near:
            return None
        if recovered:
            since = max(a.recovered_since for a in active if a.recovered_since)
            if (now - since).total_seconds() >= self.cfg.clear_min_seconds:
                return f"clearing report, anomaly recovered, {detail}"
            return None
        ends = [a.ended_at for a in inc.anomalies.values() if a.traffic and a.ended_at]
        since_anomaly = (now - max(ends)).total_seconds() if ends else float("inf")
        if inc.cleared_at is not None and since_anomaly >= self.cfg.clear_min_seconds:
            return f"clearing report, anomaly over, {detail}"
        quiet_since = inc.last_active_at or inc.suspected_at
        if (now - quiet_since).total_seconds() >= self.cfg.quiet_seconds:
            return f"quiet for {self.cfg.quiet_seconds:.0f}s, anomaly over, {detail}"
        return None

    def _baseline(self, camera_id: str, now: datetime) -> dict | None:
        q = self.metrics.get(camera_id) or []
        cutoff = now - timedelta(seconds=self.cfg.metrics_recent_seconds)
        past = [p for p in q if now - self.window <= p.ts < cutoff]
        speeds = [p.avg_speed for p in past if p.avg_speed is not None]
        if len(speeds) < 6:
            return None
        return {
            "avg_speed_kmh": round(statistics.median(speeds), 2),
            "stopped_vehicles": statistics.median(p.stopped for p in past),
        }

    def _near_baseline(self, inc: FusedIncident, now: datetime) -> tuple[bool, str]:
        q = self.metrics.get(inc.camera_id) or []
        since = now - timedelta(seconds=self.cfg.metrics_recent_seconds)
        recent = [p for p in q if p.ts >= since]
        if not inc.baseline or not recent:
            return True, "no camera metrics to compare"
        speeds = [p.avg_speed for p in recent if p.avg_speed is not None]
        speed = statistics.fmean(speeds) if speeds else None
        stopped = statistics.fmean(p.stopped for p in recent)
        base = inc.baseline
        speed_ok = speed is None or speed >= self.cfg.near_baseline_ratio * base["avg_speed_kmh"]
        stopped_ok = stopped <= base["stopped_vehicles"] + 1
        detail = f"camera speed {speed or 0:.1f} vs baseline {base['avg_speed_kmh']:.1f} km/h"
        return speed_ok and stopped_ok, detail

    # ------------------------------------------------------------------ output

    def _emit(self, inc: FusedIncident) -> None:
        key = str(inc.incident_id)
        reasons = self._reasons.pop(key, [])
        evidence = self._pending.pop(key, [])
        if not reasons and not evidence:
            return
        last = inc.published
        meaningful = (
            not last
            or last.get("status") != inc.status
            or last.get("severity") != inc.severity
            or last.get("incident_type") != inc.incident_type
            or last.get("vision_degraded") != inc.vision_degraded
            or abs(last.get("fused", 0.0) - inc.fused) >= self.cfg.publish_confidence_delta
        )
        if meaningful:
            inc.version += 1
            inc.published = {
                "status": inc.status,
                "severity": inc.severity,
                "incident_type": inc.incident_type,
                "fused": inc.fused,
                "vision_degraded": inc.vision_degraded,
            }
            inc.history.append(
                {
                    "version": inc.version,
                    "status": inc.status,
                    "fused": inc.fused,
                    "severity": inc.severity,
                    "incident_type": inc.incident_type,
                    "reasons": reasons,
                }
            )
        self._changes.append(Change(inc, reasons, meaningful, evidence))
        if inc.status == "resolved":
            self._close(inc)

    def _close(self, inc: FusedIncident) -> None:
        """Forget a resolved incident and the evidence it used, so it can't re-trigger."""
        self.incidents.pop(str(inc.incident_id), None)
        pool = self.reports.get(inc.camera_id, {})
        others = {k for o in self.open_on(inc.camera_id) for k in o.reports}
        end = inc.end_time or datetime.max.replace(tzinfo=inc.suspected_at.tzinfo)
        for key, r in list(pool.items()):
            if key in others:
                continue
            fits = (
                type_compatibility(inc.incident_type, r.incident_type) >= self.cfg.min_compatibility
            )
            if key in inc.reports or (r.ts <= end and fits):
                del pool[key]
        for key in list(inc.anomalies):
            self.anomalies.pop(key, None)

    def _expire(self, now: datetime) -> None:
        horizon = now - self.window
        for pool in self.reports.values():
            for key in [k for k, r in pool.items() if r.ts < horizon]:
                del pool[key]
        for key in [k for k, r in self.unlocated.items() if r.ts < horizon]:
            del self.unlocated[key]
        open_ids = {k for inc in self.incidents.values() for k in inc.anomalies}
        for key in [
            k
            for k, a in self.anomalies.items()
            if not a.active and (a.ended_at or now) < horizon and k not in open_ids
        ]:
            del self.anomalies[key]

    # ------------------------------------------------------------------ restart recovery

    def restore(
        self,
        incidents: list[FusedIncident],
        anomalies: list[AnomalyEvidence],
        reports: list[ReportEvidence],
        system: dict | None,
        metrics: list[tuple[str, MetricPoint]] | None = None,
    ) -> None:
        """Rebuild state from Postgres: open incidents (with their evidence) and the recent
        pools. Evidence already attached is keyed by id, so redelivered messages after a
        restart attach to the same incident instead of creating a duplicate."""
        for inc in incidents:
            self.incidents[str(inc.incident_id)] = inc
            for a in inc.anomalies.values():
                self.anomalies[a.anomaly_id] = a
        for a in anomalies:
            self.anomalies.setdefault(a.anomaly_id, a)
            for inc in self.open_on(a.camera_id):
                if a.anomaly_id in inc.anomalies:
                    inc.anomalies[a.anomaly_id] = self.anomalies[a.anomaly_id]
        for ev in reports:
            if ev.camera_id is None:
                if ev.places and ev.stage == "active":
                    self.unlocated.setdefault(ev.report_id, ev)
            else:
                self.reports.setdefault(ev.camera_id, {}).setdefault(ev.report_id, ev)
        for camera_id, point in metrics or []:  # oldest first
            self.metrics.setdefault(camera_id, deque(maxlen=400)).append(point)
        self.system = system
