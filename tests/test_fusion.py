"""fusion_service: type mapping, confidence components, window grouping, lifecycle, clearing
reports, unlocated reports, system degradation (engine only, no I/O), and restart recovery
against a scratch database."""

import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "fusion_service")]

pytest.importorskip("rapidfuzz")

from fusion_service import confidence  # noqa: E402
from fusion_service.config import FusionConfig  # noqa: E402
from fusion_service.engine import FusionEngine  # noqa: E402
from fusion_service.model import AnomalyEvidence, ReportEvidence  # noqa: E402
from fusion_service.types import (  # noqa: E402
    hint_agreement,
    robust_text_severity,
    type_compatibility,
    type_from_hints,
    vision_severity,
    vote_type,
)
from traffic_common.schemas import Anomaly, Entity, NLPResult, TrafficMetrics  # noqa: E402

T0 = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)
CAM = "cam-old-street"
SOURCES = ["citizen", "social_media", "operator", "police"]


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def anomaly(
    t: float,
    event: str = "start",
    hints=("speed_drop", "stopped_vehicles"),
    ratio: float = 3.0,
    camera: str = CAM,
    anomaly_id: uuid.UUID | None = None,
    severity: str = "high",
    started: float | None = None,
) -> Anomaly:
    ts = at(t)
    return Anomaly(
        anomaly_id=anomaly_id or uuid.uuid4(),
        camera_id=camera,
        event=event,
        severity=severity,
        anomaly_type=hints[0] if hints else "unknown",
        hints=list(hints),
        metric_name="loop_speed_kmh_rel",
        observed_value=0.2,
        score=ratio,
        threshold=1.0,
        model_name="m",
        model_version="1",
        started_at=at(started if started is not None else t - 10),
        ended_at=ts if event == "end" else None,
        window_start=ts - timedelta(seconds=60),
        window_end=ts,
        timestamp=ts,
        congestion_level="heavy",
    )


def system(t: float, event: str = "start", cameras=(CAM,)) -> Anomaly:
    ts = at(t)
    return Anomaly(
        anomaly_id=uuid.UUID(int=7),
        scope="system",
        affected_cameras=list(cameras),
        event=event,
        severity="high",
        anomaly_type="camera_degraded",
        hints=["camera_degraded"],
        metric_name="frames",
        observed_value=0.5,
        score=4,
        threshold=3,
        model_name="system-health-rules",
        model_version="1",
        started_at=at(0),
        ended_at=ts if event == "end" else None,
        window_start=ts - timedelta(seconds=15),
        window_end=ts,
        timestamp=ts,
    )


def report(
    t: float,
    incident_type: str = "accident",
    severity: str = "high",
    source: str = "citizen",
    camera: str | None = CAM,
    stage: str = "active",
    place: str | None = None,
    confidence_: float = 0.95,
) -> NLPResult:
    text_place = place or "Old Street"
    return NLPResult(
        report_id=uuid.uuid4(),
        source=source,
        incident_type=incident_type,
        incident_confidence=confidence_,
        severity=severity,
        severity_confidence=0.9,
        stage=stage,
        entities=[Entity(text=text_place, label="LOCATION", start=0, end=len(text_place))],
        camera_id=camera,
        camera_confidence=0.95 if camera else None,
        model_name="incident-text-classifier",
        model_version="1",
        timestamp=at(t),
    )


def burst(engine: FusionEngine, t: float, n: int = 4, incident_type: str = "accident", **kw):
    for i in range(n):
        engine.on_nlp(report(t + i, incident_type, source=SOURCES[i % 4], **kw), at(t + i))


def published(engine: FusionEngine) -> list:
    return [(c.incident.status, c.incident.version, c.reasons) for c in engine.drain() if c.publish]


def only(engine: FusionEngine, camera: str = CAM):
    """The camera's single open incident."""
    [inc] = engine.open_on(camera)
    return inc


@pytest.fixture
def engine() -> FusionEngine:
    names = [
        "cam-old-street",
        "cam-hyde-park-corner",
        "cam-elephant-castle",
        "cam-trafalgar-square",
    ]
    return FusionEngine(FusionConfig(), {c: c for c in names})


# ----------------------------------------------------------------------------- types


def test_type_mapping_from_hints_and_reports() -> None:
    assert type_from_hints(["speed_drop", "stopped_vehicles"]) == "accident"
    assert type_from_hints(["stopped_vehicles"]) == "breakdown"
    assert type_from_hints(["speed_drop"]) == "congestion"
    assert type_from_hints(["pedestrian_in_road"]) == "police_activity"
    assert type_from_hints(["camera_degraded", "sensor_dropout"]) is None  # equipment
    # Congestion is usually the consequence: it only wins with a clear majority.
    assert vote_type({"breakdown": 1.8, "congestion": 2.4}) == "breakdown"
    assert vote_type({"breakdown": 0.9, "congestion": 2.4}) == "congestion"
    assert vote_type({}) is None


def test_agreement_and_compatibility() -> None:
    assert hint_agreement(["speed_drop", "stopped_vehicles"], "accident") == 1.0
    assert hint_agreement(["speed_drop"], "weather_hazard") == 0.8
    assert hint_agreement(["camera_degraded"], "accident") == 0.0
    assert hint_agreement(["speed_drop"], None) == 0.0
    assert type_compatibility("accident", "accident") == 1.0
    assert type_compatibility("road_works", "congestion") == 0.7
    assert type_compatibility("accident", "breakdown") == 0.6
    assert type_compatibility("road_works", "weather_hazard") < 0.5


def test_severity_rules() -> None:
    assert robust_text_severity(["critical", "high", "high", "low"], 2) == "high"
    assert robust_text_severity(["critical", "critical", "high"], 2) == "critical"
    assert robust_text_severity(["medium"], 2) == "medium"
    assert vision_severity("critical", "heavy") == "high"  # critical needs a standstill
    assert vision_severity("critical", "standstill") == "critical"
    assert vision_severity("low", "standstill") == "high"
    assert vision_severity(None, None) is None


# ----------------------------------------------------------------------------- confidence


def test_confidence_components_are_explainable() -> None:
    cfg = FusionConfig()
    anomalies = [
        AnomalyEvidence("a", CAM, at(0), at(60), peak_ratio=2.5, hints=["speed_drop"], active=True)
    ]
    v = confidence.vision_component(anomalies, at(60), cfg)
    assert v["ratio_score"] == pytest.approx(0.5)  # (2.5 - 1) / (4 - 1)
    assert v["duration_score"] == pytest.approx(0.5)  # 60 s of 120
    assert v["score"] == pytest.approx(0.3 + 0.7 * (0.6 * 0.5 + 0.4 * 0.5))
    reports = [
        ReportEvidence(f"r{i}", CAM, at(i), "accident", 0.9, "high", s, 1.0)
        for i, s in enumerate(["citizen", "police", "citizen"])
    ] + [ReportEvidence("rc", CAM, at(5), "congestion", 0.8, "medium", "operator", 1.0)]
    t = confidence.text_component(reports, at(10), cfg)
    assert t["type"] == "accident" and t["supporting_reports"] == 3.5  # congestion counts half
    later = confidence.text_component(reports, at(600), cfg)  # same reports, 10 min on
    assert later["recent_supporting_reports"] == 0 and later["score"] < t["score"]
    assert t["n_sources"] == 3 and t["source_score"] == 1.0
    a = confidence.agreement_component(v, t)
    assert a["score"] == 0.8 and a["both_modalities"]
    f = confidence.fuse(v, t, a, cfg)
    primary, secondary = max(v["score"], t["score"]), min(v["score"], t["score"])
    assert f["fused"] == pytest.approx(0.6 * primary + 0.25 * secondary + 0.15 * 0.8, abs=1e-3)
    # Unrelated reports are attached but add nothing.
    noise = reports + [ReportEvidence("rx", CAM, at(6), "road_works", 0.99, "low", "news", 1.0)]
    assert confidence.text_component(noise, at(10), cfg)["score"] == t["score"]
    degraded = [AnomalyEvidence("a", CAM, at(0), at(60), 2.5, ["speed_drop"], degraded=True)]
    dv = confidence.vision_component(degraded, at(60), cfg)
    assert dv["degraded"] and dv["score"] == pytest.approx(v["score"] * 0.3, abs=1e-3)


# ----------------------------------------------------------------------------- lifecycle


def test_vision_suspects_text_confirms_then_quiet_resolves(engine) -> None:
    aid = uuid.uuid4()
    engine.on_anomaly(anomaly(0, anomaly_id=aid), at(0))
    [(status, version, reasons)] = published(engine)
    assert (status, version) == ("suspected", 1)
    assert reasons[:2] == ["created: traffic anomaly", "anomaly attached"]
    inc = only(engine)
    assert inc.incident_type == "accident" and inc.status == "suspected"
    burst(engine, 30)  # 4 agreeing accident reports from 4 sources
    changes = published(engine)
    assert changes[-1][0] == "confirmed" and inc.at_confirmation["rule"] == "both modalities agree"
    assert inc.at_confirmation["components"]["agreement"]["score"] == 1.0
    engine.on_anomaly(anomaly(120, "end", anomaly_id=aid), at(120))
    engine.tick(at(200))
    assert inc.status == "confirmed"  # anomaly over, but not quiet long enough yet
    # Quiet counts from the last time it was firing: the report burst (30-33 s) still counted
    # as recent at 200 s. Sparse same-type background reports don't keep it alive.
    for t in (260, 380):
        engine.on_nlp(report(t, source="news"), at(t))
    engine.tick(at(480))
    assert inc.status == "confirmed"
    engine.tick(at(200 + 300))
    assert inc.status == "resolved" and "quiet" in inc.resolution
    assert not engine.open_on(CAM)
    versions = [v for _, v, _ in published(engine)]
    assert versions == sorted(versions)


def test_small_changes_are_persisted_but_not_published(engine) -> None:
    burst(engine, 0, n=8)  # strong text: confirmed
    engine.drain()
    inc = only(engine)
    version = inc.version
    engine.on_nlp(report(20, source="news"), at(20))  # confidence moves < 0.1
    [change] = engine.drain()
    assert not change.publish and change.new_evidence and inc.version == version


def test_clearing_report_speeds_up_resolution(engine) -> None:
    aid = uuid.uuid4()
    engine.on_anomaly(anomaly(0, anomaly_id=aid), at(0))
    burst(engine, 10)
    engine.on_nlp(report(60, stage="cleared", source="operator"), at(60))
    assert only(engine).status == "confirmed"  # anomaly still active
    engine.on_anomaly(anomaly(80, "end", anomaly_id=aid), at(80))
    engine.tick(at(95))
    assert only(engine).status == "confirmed"  # within clear_min_seconds
    engine.tick(at(101))
    assert not engine.open_on(CAM)  # resolved 21 s after the anomaly, not 300 s
    changes = engine.drain()
    assert "clearing report" in changes[-1].incident.resolution


def test_window_groups_evidence_and_old_reports_stay_out(engine) -> None:
    engine.on_nlp(report(0, source="citizen"), at(0))  # too old to join when it starts
    burst(engine, 700)
    inc = only(engine)
    assert len(inc.reports) == 4  # the report 700 s earlier is outside the 600 s window
    engine.on_anomaly(anomaly(760), at(760))  # 1 min later: same incident
    assert len(inc.anomalies) == 1 and only(engine) is inc
    other = "cam-hyde-park-corner"
    engine.on_anomaly(anomaly(770, camera=other), at(770))
    assert only(engine, other) is not inc  # per camera


def test_long_running_anomaly_does_not_create_an_incident(engine) -> None:
    aid = uuid.uuid4()
    engine.on_anomaly(anomaly(0, anomaly_id=aid, started=0), at(0))
    engine.drain()
    engine.incidents.clear()  # e.g. fusion started long after the alert began
    engine.on_anomaly(anomaly(3600, "update", anomaly_id=aid, started=0), at(3600))
    assert engine.incidents == {}


def test_unrelated_burst_gets_its_own_incident_on_the_same_camera(engine) -> None:
    burst(engine, 0, incident_type="accident")  # e.g. a false alarm from chatter
    accident = only(engine)
    burst(engine, 60, incident_type="road_works", severity="medium")
    opens = engine.open_on(CAM)
    assert [i.incident_type for i in opens] == ["accident", "road_works"]
    works = opens[1]
    assert all(r.incident_type == "road_works" for r in works.reports.values())
    assert all(r.incident_type == "accident" for r in accident.reports.values())
    engine.on_nlp(report(70, "accident", source="police"), at(70))
    assert len(accident.reports) == 5 and len(works.reports) == 4  # routed by type
    engine.on_nlp(report(90, "road_works", stage="cleared", source="operator"), at(90))
    assert works.status == "resolved" and accident.status != "resolved"
    assert engine.open_on(CAM) == [accident]


def test_unknown_cameras_are_ignored(engine) -> None:
    engine.on_anomaly(anomaly(0, camera="test-cam-x"), at(0))
    burst(engine, 0, camera="test-cam-x")
    assert engine.incidents == {} and engine.ignored == 5


def test_background_chatter_does_not_create_incidents(engine) -> None:
    for i, t in enumerate(["accident", "road_works", "debris", "breakdown", "accident"]):
        engine.on_nlp(report(i * 20, t, source=SOURCES[i % 4]), at(i * 20))
    assert engine.incidents == {}  # no 4 same-type reports
    for i in range(4):
        engine.on_nlp(report(100 + i, "weather_hazard", source="citizen"), at(100 + i))
    assert engine.incidents == {}  # one source only
    engine.on_anomaly(anomaly(200, hints=("camera_degraded",)), at(200))
    assert engine.incidents == {}  # equipment fault, not an incident


def test_text_only_incident_resolves_on_clearing_report(engine) -> None:
    burst(engine, 0, incident_type="road_works", severity="medium")
    inc = only(engine)
    assert inc.incident_type == "road_works" and inc.status == "suspected"
    assert inc.severity == "medium"
    engine.on_nlp(report(90, "road_works", stage="cleared", source="operator"), at(90))
    assert inc.status == "resolved" and "clearing report" in inc.resolution


def test_unlocated_report_attaches_by_place(engine) -> None:
    unlocated = report(5, camera=None, place="brook drive", source="police")
    engine.on_nlp(unlocated, at(5))
    assert str(unlocated.report_id) in engine.unlocated
    burst(engine, 10, place="Brook Drive")  # located reports that also name Brook Drive
    inc = only(engine)
    ev = inc.reports[str(unlocated.report_id)]
    assert ev.unlocated and ev.camera_id == CAM
    assert ev.camera_confidence == pytest.approx(0.8)
    assert inc.components["text"]["unlocated_reports"] == 1
    far = report(20, camera=None, place="Kings Cross")
    engine.on_nlp(far, at(20))
    assert str(far.report_id) not in inc.reports


def test_system_degradation_blocks_vision_only_and_degrades_evidence(engine) -> None:
    aid = uuid.uuid4()
    engine.on_anomaly(anomaly(0, anomaly_id=aid), at(0))
    before = only(engine).components["vision"]["score"]
    engine.drain()
    engine.on_anomaly(system(30, cameras=[CAM, "cam-elephant-castle"]), at(30))
    inc = only(engine)
    assert inc.vision_degraded and inc.components["vision"]["degraded"]
    assert inc.components["vision"]["score"] < before
    [change] = engine.drain()
    assert change.publish and "vision degraded (system-wide degradation)" in change.reasons
    # A new vision-only anomaly on an affected camera during the degradation: no incident.
    engine.on_anomaly(anomaly(40, camera="cam-elephant-castle"), at(40))
    assert not engine.open_on("cam-elephant-castle")
    # Text still creates incidents there.
    burst(engine, 50, camera="cam-elephant-castle")
    assert only(engine, "cam-elephant-castle").status == "suspected"
    # The degraded anomaly is still "active", but it can't hold the incident open.
    assert not only(engine).active_anomaly
    engine.on_anomaly(system(90, "end"), at(90))
    engine.on_anomaly(anomaly(100, camera="cam-trafalgar-square"), at(100))
    assert engine.open_on("cam-trafalgar-square")  # after it ends, vision creates again


def test_strong_vision_alone_confirms(engine) -> None:
    aid = uuid.uuid4()
    engine.on_anomaly(anomaly(0, ratio=9.0, anomaly_id=aid, started=0), at(0))
    engine.tick(at(130))  # 130 s at 9x threshold
    inc = only(engine)
    assert inc.status == "confirmed" and inc.at_confirmation["rule"] == "strong vision"


def test_metrics_must_be_back_near_baseline(engine) -> None:
    for i in range(60):  # 5 minutes at 30 km/h
        engine.on_metrics(metrics_at(i * 5 - 300, 30.0))
    aid = uuid.uuid4()
    engine.on_anomaly(anomaly(0, anomaly_id=aid), at(0))
    inc = only(engine)
    assert inc.baseline["avg_speed_kmh"] == 30.0
    for i in range(100):
        engine.on_metrics(metrics_at(i * 5, 8.0))  # camera still slow for 500 s
    engine.on_anomaly(anomaly(100, "end", anomaly_id=aid), at(100))
    engine.tick(at(100 + 400))
    assert inc.status != "resolved"
    for i in range(14):
        engine.on_metrics(metrics_at(500 + i * 5, 29.0))
    engine.tick(at(570))
    assert inc.status == "resolved"


def metrics_at(t: float, speed: float) -> TrafficMetrics:
    end = at(t)
    return TrafficMetrics(
        camera_id=CAM,
        window_start=end - timedelta(seconds=5),
        window_end=end,
        vehicle_count=5,
        avg_speed_kmh=speed,
        timestamp=end,
    )


# ----------------------------------------------------------------------------- persistence


@pytest.fixture
def db(scratch_database: str):
    from alembic import command

    from traffic_common.db import create_db_engine
    from traffic_common.db.migrate import alembic_config
    from traffic_common.db.seed import seed_cameras

    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    seed_cameras(engine)
    yield engine
    engine.dispose()


def store_report(db, result: NLPResult, text: str = "crash at Old Street") -> None:
    from sqlalchemy import insert

    from traffic_common.db.models import Report

    with db.begin() as conn:
        conn.execute(
            insert(Report).values(
                id=result.report_id, source=result.source, text=text, reported_at=result.timestamp
            )
        )


def test_restart_recovers_open_incidents_without_duplicates(db) -> None:
    from sqlalchemy import text

    from fusion_service.service import FusionService

    now = {"t": at(0)}
    sent: list = []
    first = FusionService(FusionConfig(), db, lambda t, m: sent.append(m), clock=lambda: now["t"])
    reports = [report(i, source=SOURCES[i % 4]) for i in range(5)]
    for r in reports:
        store_report(db, r)
        now["t"] = r.timestamp
        first.handle(r)
    assert [m.status for m in sent] == ["suspected", "confirmed"]
    incident_id = sent[0].incident_id
    with db.connect() as conn:
        n_evidence = conn.scalar(text("SELECT count(*) FROM incident_evidence"))
    assert n_evidence == 5

    # "Restart": a new service rebuilds from Postgres; the last report is redelivered.
    second = FusionService(FusionConfig(), db, lambda t, m: sent.append(m), clock=lambda: now["t"])
    assert second.recover() == 1
    restored = only(second.fusion)
    assert restored.incident_id == incident_id and restored.status == "confirmed"
    assert restored.version == sent[-1].version
    second.handle(reports[-1])  # redelivered: already attached
    extra = report(30, source="police")
    store_report(db, extra)
    now["t"] = extra.timestamp
    second.handle(extra)
    second.handle(report(31, stage="cleared", source="operator"))
    assert sent[-1].incident_id == incident_id and sent[-1].status == "resolved"
    assert sent[-1].version == max(m.version for m in sent)  # versions continue
    with db.connect() as conn:
        rows = conn.execute(text("SELECT id, status, version FROM incidents")).all()
        n_evidence = conn.scalar(text("SELECT count(*) FROM incident_evidence"))
    assert [(r.id, r.status) for r in rows] == [(incident_id, "resolved")]
    assert n_evidence == 6  # the redelivered report wasn't linked twice


def test_evidence_waits_for_its_target_row(db) -> None:
    from sqlalchemy import text

    from fusion_service.service import FusionService

    sent: list = []
    svc = FusionService(FusionConfig(), db, lambda t, m: sent.append(m), clock=lambda: at(0))
    for i in range(4):
        svc.handle(report(i, source=SOURCES[i]))  # their report rows don't exist yet
    assert len(svc.store.pending) == 4
    with db.connect() as conn:
        assert conn.scalar(text("SELECT count(*) FROM incident_evidence")) == 0
    # The rows appear later (as when nlp_service stores after publishing): linked on a tick.
    for change in only(svc.fusion).reports.values():
        store_report(
            db,
            NLPResult(
                report_id=change.report_id,
                source=change.source,
                incident_type="accident",
                incident_confidence=0.9,
                model_name="m",
                model_version="1",
                timestamp=change.ts,
            ),
        )
    svc.store.flush_evidence()
    assert svc.store.pending == []
    with db.connect() as conn:
        assert conn.scalar(text("SELECT count(*) FROM incident_evidence")) == 4
