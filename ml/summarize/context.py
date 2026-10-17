"""The facts a summary may use, and nothing else. ``facts_text()`` is both the model's input
and what the faithfulness guard checks the output against."""

import os
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import text

# Clock times in summaries are written in this zone, with its abbreviation ("18:48 BST"), the
# same zone the dashboard displays times in. The demo cameras are in London.
DISPLAY_TIMEZONE = os.getenv("DISPLAY_TIMEZONE", "").strip() or "Europe/London"
TZ = ZoneInfo(DISPLAY_TIMEZONE)


def local_time(at: datetime, tz: ZoneInfo = TZ) -> str:
    """ "18:48 BST": 24-hour time in the display zone with its abbreviation."""
    t = at.astimezone(tz)
    return f"{t:%H:%M} {t.tzname()}"


TYPE_LABEL = {
    "accident": "accident",
    "breakdown": "broken-down vehicle",
    "road_works": "road works",
    "congestion": "congestion",
    "weather_hazard": "weather hazard",
    "debris": "debris in the road",
    "police_activity": "police activity",
    "none": "incident",
}
SOURCE_LABEL = {
    "citizen": "drivers",
    "social_media": "social media",
    "police": "police",
    "operator": "the road operator",
    "news": "news",
}


# Camera and loop speed ratios (percent of their baselines) this close count as agreeing.
CAMERA_AGREEMENT_POINTS = 25


@dataclass
class IncidentContext:
    incident_id: str
    version: int
    camera_id: str
    location: str  # camera name, e.g. "Hyde Park Corner - Piccadilly approach"
    area: str  # e.g. "Hyde Park Corner, London W1J"
    incident_type: str
    severity: str
    status: str  # suspected | confirmed | resolved
    start_time: datetime
    reference_time: datetime  # when this version was published
    congestion: str | None = None
    speed_kmh: float | None = None  # camera (video) speed, recent mean
    baseline_speed_kmh: float | None = None  # camera speed before the incident
    loop_speed_kmh: float | None = None  # loop detectors, recent mean
    loop_baseline_kmh: float | None = None  # loop detectors before the incident
    stopped_vehicles: int | None = None
    pedestrians_in_roadway: int | None = None
    n_reports: int = 0
    sources: list[str] = field(default_factory=list)
    vehicles: list[str] = field(default_factory=list)
    lanes: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)  # place names the reports mention
    fused_confidence: float = 0.0
    vision_score: float = 0.0
    text_score: float = 0.0
    agreement_score: float = 0.0
    vision_present: bool = False
    vision_degraded: bool = False
    cleared_reports: int = 0

    # ------------------------------------------------------------------ derived facts

    @property
    def type_label(self) -> str:
        return TYPE_LABEL.get(self.incident_type, self.incident_type.replace("_", " "))

    @property
    def short_location(self) -> str:
        return self.location.split(" - ")[0].strip()

    @property
    def start_local(self) -> str:
        return local_time(self.start_time)

    @property
    def duration_min(self) -> int:
        return max(0, round((self.reference_time - self.start_time).total_seconds() / 60))

    @property
    def speed_pct(self) -> int | None:
        if self.speed_kmh is None or not self.baseline_speed_kmh:
            return None
        return round(100 * self.speed_kmh / self.baseline_speed_kmh)

    @property
    def loop_pct(self) -> int | None:
        if self.loop_speed_kmh is None or not self.loop_baseline_kmh:
            return None
        return round(100 * self.loop_speed_kmh / self.loop_baseline_kmh)

    @property
    def camera_agrees(self) -> bool:
        """Camera and loop detectors tell the same story (relative to their baselines)."""
        if self.speed_pct is None or self.loop_pct is None:
            return False
        return abs(self.speed_pct - self.loop_pct) <= CAMERA_AGREEMENT_POINTS

    @property
    def show_camera_speed(self) -> bool:
        """Loop detectors measure the carriageway directly; the camera's speed is only worth
        stating when it agrees with them or when there is no loop data (the footage may not
        show the incident at all)."""
        return self.speed_pct is not None and (self.loop_pct is None or self.camera_agrees)

    @property
    def source_labels(self) -> list[str]:
        return [SOURCE_LABEL.get(s, s.replace("_", " ")) for s in self.sources]

    @property
    def recommended_action(self) -> str:
        """Deterministic, from status and severity (the model must not invent one)."""
        if self.status == "resolved":
            return "monitor for residual queues; no further action needed"
        if self.status == "suspected":
            return "verify on the camera feed before acting"
        if self.severity == "critical":
            return "dispatch emergency response and divert traffic"
        if self.severity == "high":
            return "divert traffic and dispatch a patrol to assess"
        if self.severity == "medium":
            return "monitor and prepare a diversion"
        return "monitor"

    def facts_text(self) -> str:
        """One fact per line: the model's input and the faithfulness reference."""
        lines = [
            f"Location: {self.location} ({self.area})",
            f"Incident: {self.type_label}, severity {self.severity}, status {self.status}",
            f"Started: {self.start_local}, {self.duration_min} minutes ago",
        ]
        traffic = []
        if self.congestion:
            traffic.append(f"congestion {self.congestion.replace('_', ' ')}")
        if self.loop_pct is not None:
            traffic.append(
                f"loop detector speed {self.loop_speed_kmh:.0f} km/h against "
                f"{self.loop_baseline_kmh:.0f} km/h normally ({self.loop_pct} percent)"
            )
        if self.show_camera_speed:
            traffic.append(
                f"camera speed {self.speed_kmh:.0f} km/h against {self.baseline_speed_kmh:.0f} "
                f"km/h normally ({self.speed_pct} percent)"
            )
        if self.stopped_vehicles:
            traffic.append(f"{self.stopped_vehicles} stopped vehicles")
        if self.pedestrians_in_roadway:
            traffic.append(f"{self.pedestrians_in_roadway} pedestrians in the roadway")
        if traffic:
            lines.append("Traffic: " + "; ".join(traffic))
        if self.n_reports:
            lines.append(
                f"Reports: {self.n_reports} from {len(self.sources)} sources "
                f"({', '.join(self.source_labels)})"
            )
        if self.vehicles:
            lines.append("Vehicles mentioned: " + ", ".join(self.vehicles))
        if self.lanes:
            lines.append("Lanes mentioned: " + ", ".join(self.lanes))
        if self.places:
            lines.append("Places mentioned: " + ", ".join(self.places))
        evidence = []
        if self.vision_present:
            evidence.append(
                "traffic anomaly on the camera"
                + (" (vision degraded)" if self.vision_degraded else "")
            )
        if self.n_reports:
            evidence.append("text reports")
        if evidence:
            lines.append(
                f"Evidence: {' and '.join(evidence)}; confidence {self.fused_confidence:.2f}"
            )
        lines.append(f"Recommended action: {self.recommended_action}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_time"] = self.start_time.isoformat()
        d["reference_time"] = self.reference_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "IncidentContext":
        d = dict(d)
        d["start_time"] = datetime.fromisoformat(d["start_time"])
        d["reference_time"] = datetime.fromisoformat(d["reference_time"])
        return cls(**d)


# ----------------------------------------------------------------------------- from Postgres


_IRREGULAR = {
    "lorries": "lorry",
    "buses": "bus",
    "coaches": "coach",
    "taxis": "taxi",
    "minibuses": "minibus",
}


def singular(word: str) -> str:
    """lorries -> lorry, buses -> bus, vans -> van; 'bus' stays 'bus'."""
    word = word.lower().strip()
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def _top(values: list[str], n: int = 3) -> list[str]:
    return [v for v, _ in Counter(values).most_common(n)]


def load_context(conn, incident_id, overrides: dict | None = None) -> IncidentContext | None:
    """Build the context for an incident from Postgres. ``overrides`` (from the Incident
    message being summarised: version, status, type, severity, ...) win over the row, which
    may already be a newer version."""
    row = (
        conn.execute(
            text(
                "SELECT i.*, c.name AS camera_name, c.location_name FROM incidents i "
                "JOIN cameras c ON c.id = i.camera_id WHERE i.id = :id"
            ),
            {"id": str(incident_id)},
        )
        .mappings()
        .first()
    )
    if row is None:
        return None
    o = overrides or {}
    state = (row["details"] or {}).get("state", {})
    components = o.get("confidence_components") or row["confidence_components"] or {}
    ref = o.get("timestamp") or row["updated_at"] or datetime.now(UTC)
    start = o.get("start_time") or row["start_time"]
    end = o.get("end_time") or row["end_time"]
    metric_end = end or ref
    m = conn.execute(
        text(
            "SELECT avg_speed_kmh, stopped_vehicles, pedestrians_in_roadway FROM traffic_metrics "
            "WHERE camera_id = :c AND window_end BETWEEN :a AND :b"
        ),
        {"c": row["camera_id"], "a": metric_end - timedelta(seconds=60), "b": metric_end},
    ).all()
    speeds = [r.avg_speed_kmh for r in m if r.avg_speed_kmh is not None]
    congestion = conn.scalar(
        text("SELECT level FROM camera_congestion WHERE camera_id = :c"), {"c": row["camera_id"]}
    )
    reports = state.get("reports", {})
    active = [r for r in reports.values() if r.get("stage", "active") == "active"]
    vehicles, lanes, places = [], [], []
    if active:
        ents = conn.execute(
            text("SELECT entities FROM nlp_results WHERE report_id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": [r["report_id"] for r in active]},
        ).scalars()
        for entities in ents:
            for e in entities or []:
                value = e["text"].strip().lower()
                if e["label"] == "VEHICLE":
                    vehicles.append(singular(value))
                elif e["label"] == "LANE":
                    lanes.append(value)
                elif e["label"] == "LOCATION":
                    places.append(e["text"].strip())
    baseline = (state.get("baseline") or {}).get("avg_speed_kmh")
    loop = (
        conn.execute(
            text(
                "SELECT avg(avg_speed_kmh) FILTER (WHERE reading_at BETWEEN :b0 AND :b1) AS base, "
                "count(avg_speed_kmh) FILTER (WHERE reading_at BETWEEN :b0 AND :b1) AS n_base, "
                "avg(avg_speed_kmh) FILTER (WHERE reading_at BETWEEN :r0 AND :r1) AS now "
                "FROM sensor_readings WHERE camera_id = :c AND reading_at BETWEEN :b0 AND :r1"
            ),
            {
                "c": row["camera_id"],
                # before the incident: the 10 minutes up to a minute before it started
                "b0": start - timedelta(minutes=10),
                "b1": start - timedelta(minutes=1),
                # now: the last 90 seconds (readings arrive every 15 s)
                "r0": metric_end - timedelta(seconds=90),
                "r1": metric_end,
            },
        )
        .mappings()
        .first()
    )
    loop_base = float(loop["base"]) if loop and loop["base"] and loop["n_base"] >= 3 else None
    loop_now = float(loop["now"]) if loop and loop["now"] is not None else None
    vision = components.get("vision", {})
    text_c = components.get("text", {})
    return IncidentContext(
        incident_id=str(row["id"]),
        version=int(o.get("version") or row["version"]),
        camera_id=row["camera_id"],
        location=row["camera_name"],
        area=row["location_name"],
        incident_type=str(o.get("incident_type") or row["incident_type"]),
        severity=str(o.get("severity") or row["severity"]),
        status=str(o.get("status") or row["status"]),
        start_time=start,
        reference_time=end or ref,
        congestion=congestion,
        speed_kmh=round(statistics.fmean(speeds), 1) if speeds else None,
        baseline_speed_kmh=baseline,
        loop_speed_kmh=round(loop_now, 1) if loop_now is not None and loop_base else None,
        loop_baseline_kmh=round(loop_base, 1) if loop_now is not None and loop_base else None,
        stopped_vehicles=max((r.stopped_vehicles for r in m), default=None),
        pedestrians_in_roadway=max((r.pedestrians_in_roadway for r in m), default=None),
        n_reports=len(active),
        sources=sorted({r.get("source") or "unknown" for r in active}),
        vehicles=_top(vehicles),
        lanes=_top(lanes, 2),
        places=_top([p for p in places if p.lower() not in row["camera_name"].lower()], 3),
        fused_confidence=float(o.get("fused_confidence") or row["fused_confidence"] or 0.0),
        vision_score=float(vision.get("score", 0.0)),
        text_score=float(text_c.get("score", 0.0)),
        agreement_score=float(components.get("agreement", {}).get("score", 0.0)),
        vision_present=bool(vision.get("present")),
        vision_degraded=bool(o.get("vision_degraded", row["vision_degraded"])),
        cleared_reports=len(reports) - len(active),
    )
