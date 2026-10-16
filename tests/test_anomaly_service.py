"""anomaly_service: model reload/fallback, the sensor join, and the scoring loop's start /
end / cooldown behaviour (with the rule-based models, against a scratch Postgres database)."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "anomaly_service")]

pd = pytest.importorskip("pandas")
pytest.importorskip("mlflow")
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST  # noqa: E402

from anomaly_service.config import AnomalyConfig  # noqa: E402
from anomaly_service.models import ModelManager  # noqa: E402
from anomaly_service.state import CameraState, sensor_point  # noqa: E402
from ml.alarm import AlarmTracker  # noqa: E402
from traffic_common.schemas import Anomaly, RawEvent, TrafficMetrics  # noqa: E402

# ----------------------------------------------------------------------------- model reload


class FakeRegistry:
    def __init__(self) -> None:
        self.aliases: dict[str, dict[str, str]] = {}
        self.fail = False

    def get_registered_model(self, name):
        if self.fail:
            raise ConnectionError("mlflow down")
        if name not in self.aliases:
            raise MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)
        return type("RM", (), {"aliases": self.aliases[name]})()

    def get_model_version(self, name, version):
        return type("MV", (), {"run_id": f"run-{name}-{version}"})()


class FakePredictor:
    def __init__(self, uri: str) -> None:
        self.uri = uri

    def predict(self, raw):
        return raw


def manager(registry: FakeRegistry, loads: list[str], changes: list) -> ModelManager:
    def loader(uri: str) -> FakePredictor:
        loads.append(uri)
        return FakePredictor(uri)

    return ModelManager(
        "http://unused",
        {"congestion": "congestion-classifier", "anomaly": "traffic-anomaly-detector"},
        "champion",
        client=registry,
        loader=loader,
        on_change=changes.append,
    )


def test_falls_back_to_rules_when_nothing_registered(caplog: pytest.LogCaptureFixture) -> None:
    loads, changes = [], []
    mm = manager(FakeRegistry(), loads, changes)
    mm.refresh()
    assert mm.get("congestion").source == "rules" and mm.get("anomaly").source == "rules"
    assert mm.get("anomaly").version == "rules-1" and loads == []
    assert "RULE-BASED fallback" in caplog.text
    caplog.clear()
    assert mm.refresh() == []  # still nothing registered: no reload, no repeated warning
    assert "RULE-BASED" not in caplog.text


def test_loads_champion_and_reloads_when_alias_moves() -> None:
    registry, loads, changes = FakeRegistry(), [], []
    mm = manager(registry, loads, changes)
    mm.refresh()
    registry.aliases["traffic-anomaly-detector"] = {"champion": "1"}
    changed = mm.refresh()
    assert [m.label for m in changed] == ["traffic-anomaly-detector/1"]
    assert (
        mm.get("anomaly").source == "mlflow"
        and mm.get("anomaly").run_id == "run-traffic-anomaly-detector-1"
    )
    assert loads == ["models:/traffic-anomaly-detector/1"]
    assert mm.refresh() == []  # same version: nothing to do
    registry.aliases["traffic-anomaly-detector"] = {"champion": "2", "old": "1"}
    mm.refresh()
    assert mm.get("anomaly").version == "2" and loads[-1] == "models:/traffic-anomaly-detector/2"
    assert [c.label for c in changes][-1] == "traffic-anomaly-detector/2"
    assert mm.get("congestion").source == "rules"  # roles are independent


def test_registry_outage_keeps_current_model() -> None:
    registry, loads, changes = FakeRegistry(), [], []
    registry.aliases["congestion-classifier"] = {"champion": "3"}
    mm = manager(registry, loads, changes)
    mm.refresh()
    registry.fail = True
    assert mm.refresh() == []
    assert mm.get("congestion").version == "3"
    assert mm.get("anomaly").source == "rules"  # had nothing yet when MLflow went away


# ----------------------------------------------------------------------------- state

T0 = datetime(2026, 10, 15, 8, 0, tzinfo=UTC)


def metrics(camera: str, end: datetime, **overrides) -> TrafficMetrics:
    values = {
        "vehicle_count": 6,
        "avg_speed_kmh": 30.0,
        "p85_speed_kmh": 38.0,
        "density": 2.0,
        "stopped_vehicles": 0,
        "pedestrians_in_roadway": 0,
        "mean_confidence": 0.6,
        "frames": 24,
    }
    values.update(overrides)
    return TrafficMetrics(
        camera_id=camera,
        window_start=end - timedelta(seconds=5),
        window_end=end,
        timestamp=end,
        **values,
    )


def loop_event(camera: str, at: datetime, speed: float = 40.0) -> RawEvent:
    return RawEvent(
        event_type="loop_detector",
        source="test",
        camera_id=camera,
        timestamp=at,
        payload={
            "vehicle_count": 5,
            "avg_speed_kmh": speed,
            "occupancy": 0.15,
            "detector_id": f"{camera}-loop-1",
            "interval_seconds": 15,
        },
    )


def test_camera_state_joins_latest_reading_at_or_before_window() -> None:
    state = CameraState("cam-a", 10, AlarmTracker())
    state.add_sensor(sensor_point(loop_event("cam-a", T0, 41.0)))
    state.add_sensor(sensor_point(loop_event("cam-a", T0 + timedelta(seconds=15), 12.0)))
    for i in range(1, 5):
        state.add_metrics(metrics("cam-a", T0 + timedelta(seconds=5 * i)))
    state.add_metrics(metrics("cam-a", T0 + timedelta(seconds=60)))  # 45 s after the last reading
    state.add_metrics(metrics("cam-a", T0 + timedelta(seconds=10)))  # out of order: ignored
    assert state.frame()["loop_speed_kmh"].tolist()[:4] == [41.0, 41.0, 12.0, 12.0]
    assert pd.isna(state.frame()["loop_speed_kmh"].iloc[4])  # older than the 30 s tolerance
    assert len(state.rows) == 5
    assert sensor_point(RawEvent(event_type="weather", source="t", camera_id="cam-a")) is None


# ----------------------------------------------------------------------------- scoring loop


@pytest.fixture
def service(scratch_database: str):
    from alembic import command

    from anomaly_service.service import AnomalyService
    from traffic_common.db import create_db_engine
    from traffic_common.db.migrate import alembic_config
    from traffic_common.db.seed import seed_cameras

    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    seed_cameras(engine)
    sent: list = []
    mm = manager(FakeRegistry(), [], [])
    mm.refresh()  # rule-based models
    cfg = AnomalyConfig(max_metric_age_seconds=10**9, cooldown_seconds=60, clear_after=3)
    svc = AnomalyService(cfg, engine, mm, lambda topic, m: sent.append((topic, m)))
    yield svc, sent, engine
    engine.dispose()


def feed(svc, camera: str, start: datetime, n: int, **overrides) -> datetime:
    """Feed n windows (with loop readings) and score after each, like the 5 s loop does."""
    t = start
    for _ in range(n):
        if (t - T0).total_seconds() % 15 == 0:
            svc.handle(loop_event(camera, t, overrides.get("avg_speed_kmh", 30.0) + 10))
        svc.handle(metrics(camera, t, **overrides))
        svc.score_all()
        t += timedelta(seconds=5)
    return t


def test_incident_produces_start_then_end_event(service) -> None:
    from sqlalchemy import text

    svc, sent, engine = service
    cam = "cam-old-street"
    t = feed(svc, cam, T0, 30)  # normal
    assert sent == []
    t = feed(svc, cam, t, 8, avg_speed_kmh=4.0, stopped_vehicles=4, density=7.0)  # incident
    starts = [m for _, m in sent if isinstance(m, Anomaly) and m.event == "start"]
    assert len(starts) == 1
    start = starts[0]
    assert start.camera_id == cam and start.score >= start.threshold == 1.0
    # Stopped vehicles show up a window before the 3-window mean speed has dropped far enough.
    assert start.anomaly_type in ("speed_drop", "stopped_vehicles")
    assert {"speed_drop", "stopped_vehicles"} <= set(start.hints)
    assert (
        start.model_name == "anomaly-rules"
        and start.model_versions["congestion-rules"] == "rules-1"
    )
    assert start.congestion_level in ("heavy", "standstill")
    t = feed(svc, cam, t, 20)  # back to normal (needs 3 checks below 0.8 x threshold)
    ends = [m for _, m in sent if isinstance(m, Anomaly) and m.event == "end"]
    assert len(ends) == 1 and ends[0].anomaly_id == start.anomaly_id
    assert ends[0].ended_at > ends[0].started_at == start.started_at
    assert ends[0].score >= start.score  # peak of the episode

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT status, ended_at, hints, anomaly_type FROM anomalies")
        ).one()
        congestion = conn.execute(
            text("SELECT level FROM camera_congestion WHERE camera_id = :c"), {"c": cam}
        ).scalar()
    assert row.status == "ended" and row.ended_at is not None
    assert row.anomaly_type == start.anomaly_type and "speed_drop" in row.hints
    assert congestion == "free_flow"


def test_cooldown_suppresses_immediate_retrigger(service) -> None:
    svc, sent, _ = service
    cam = "cam-hyde-park-corner"
    t = feed(svc, cam, T0, 30)
    t = feed(svc, cam, t, 6, avg_speed_kmh=4.0, stopped_vehicles=4)
    t = feed(svc, cam, t, 8)  # ends
    t = feed(svc, cam, t, 4, avg_speed_kmh=4.0, stopped_vehicles=4)  # within the 60 s cooldown
    assert [m.event for _, m in sent] == ["start", "end"]
    t = feed(svc, cam, t, 12)  # cooldown passes...
    feed(svc, cam, t, 6, avg_speed_kmh=4.0, stopped_vehicles=4)  # ...so this one alarms
    assert [m.event for _, m in sent] == ["start", "end", "start"]


def test_stale_metrics_are_not_scored(service) -> None:
    svc, sent, _ = service
    svc.cfg = AnomalyConfig(max_metric_age_seconds=60)
    svc.handle(metrics("cam-a", datetime.now(UTC) - timedelta(minutes=5)))
    svc.score_all()
    assert svc.stats.stale_metrics == 1 and "cam-a" not in svc.cameras and sent == []


def test_sensor_readings_are_stored(service) -> None:
    from sqlalchemy import text

    svc, _, engine = service
    event = loop_event("cam-elephant-castle", T0)
    svc.handle(event)
    svc.handle(event)  # duplicates are ignored
    svc.handle(loop_event("cam-not-in-db", T0))  # FK violation: skipped, not raised
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT camera_id, avg_speed_kmh FROM sensor_readings")).all()
    assert [(r.camera_id, r.avg_speed_kmh) for r in rows] == [("cam-elephant-castle", 40.0)]


def test_orphaned_active_anomalies_are_closed_on_startup(service) -> None:
    from sqlalchemy import text

    svc, sent, engine = service
    cam = "cam-trafalgar-square"
    t = feed(svc, cam, T0, 30)
    feed(svc, cam, t, 6, avg_speed_kmh=4.0, stopped_vehicles=4)  # starts, never ends
    assert [m.event for _, m in sent] == ["start"]
    assert svc.close_orphaned_anomalies() == 1  # what a restarted service does first
    (_, start), (_, end) = sent
    assert end.event == "end" and end.anomaly_id == start.anomaly_id  # downstream sees it close
    assert end.ended_at is not None and end.details["closed_by"] == "service restart"
    with engine.connect() as conn:
        row = conn.execute(text("SELECT status, ended_at, details FROM anomalies")).one()
    assert row.status == "ended" and row.ended_at is not None
    assert row.details["closed_by"] == "service restart" and row.details["model_source"] == "rules"
    assert svc.close_orphaned_anomalies() == 0


class Spy:
    """Wraps a predictor and remembers the frames it was given."""

    def __init__(self, inner) -> None:
        self.inner, self.frames = inner, []

    def predict(self, frame):
        self.frames.append(frame)
        return self.inner.predict(frame)


def test_frozen_baseline_is_passed_to_the_model_and_held(service) -> None:
    svc, sent, _ = service
    spy = Spy(svc.models.get("anomaly").predictor)
    svc.models.get("anomaly").predictor = spy
    cam = "cam-old-street"
    t = feed(svc, cam, T0, 40)  # normal: speed 30, loop 40
    assert "baseline_loop_speed_kmh" not in spy.frames[-1].columns
    # A long incident: the rule-based score fades as its own history fills with slow windows,
    # but speed stays far below the frozen baseline, so the hold keeps the alert open.
    t = feed(svc, cam, t, 110, avg_speed_kmh=4.0)
    assert [m.event for _, m in sent if m.event != "update"] == ["start"]
    state = svc.cameras[cam]
    assert state.frozen_baseline["avg_speed_kmh"] == pytest.approx(30.0)
    assert state.tracker.held_checks > 0
    since = spy.frames[-1][spy.frames[-1]["ts"] >= state.active.started_at]
    assert (since["baseline_avg_speed_kmh"] == 30.0).all()
    assert "camera_speed_below_0.6x_baseline" in state.active.hold_reasons
    feed(svc, cam, t, 12)  # speeds recover: the hold releases and the alert ends
    end = [m for _, m in sent if m.event == "end"]
    assert len(end) == 1 and end[0].details["held_checks"] > 0
    assert end[0].details["frozen_baseline"]["avg_speed_kmh"] == pytest.approx(30.0)
    assert state.frozen_baseline is None and state.active is None


def test_escalation_publishes_update(service) -> None:
    svc, sent, engine = service
    from sqlalchemy import text

    svc.cfg = AnomalyConfig(
        max_metric_age_seconds=10**9, severity_bounds="1.5,2.5,4", update_min_interval_seconds=0
    )
    cam = "cam-elephant-castle"
    t = feed(svc, cam, T0, 30)
    t = feed(svc, cam, t, 4, avg_speed_kmh=13.0)  # score ~1.1: start, low
    start = [m for _, m in sent if m.event == "start"]
    assert len(start) == 1 and start[0].severity == "low"
    feed(svc, cam, t, 4, avg_speed_kmh=2.0, stopped_vehicles=10)  # score ~3.3: high, new hint
    updates = [m for _, m in sent if m.event == "update"]
    assert updates and updates[-1].severity == "high"
    assert "stopped_vehicles" in updates[-1].hints
    assert updates[-1].anomaly_id == start[0].anomaly_id and updates[-1].score > start[0].score
    with engine.connect() as conn:
        row = conn.execute(text("SELECT severity, update_count, status FROM anomalies")).one()
    assert (row.severity, row.status) == ("high", "active") and row.update_count == len(updates)


def test_severity_bounds(service) -> None:
    svc, _, _ = service
    assert [svc.severity(r) for r in (1.0, 2.0, 4.0, 8.0, 20.0)] == [
        "low",
        "medium",
        "high",
        "critical",
        "critical",
    ]


# ----------------------------------------------------------------------------- system degradation

CAMERAS = ["cam-elephant-castle", "cam-hyde-park-corner", "cam-old-street", "cam-trafalgar-square"]


def feed_all(svc, start: datetime, n: int, per_camera: dict | None = None, **overrides) -> datetime:
    """Feed n windows to every demo camera, scoring once per window (one check for all)."""
    t = start
    for _ in range(n):
        for cam in CAMERAS:
            values = {**overrides, **(per_camera or {}).get(cam, {})}
            if (t - T0).total_seconds() % 15 == 0:
                svc.handle(loop_event(cam, t, values.get("avg_speed_kmh", 30.0) + 10))
            svc.handle(metrics(cam, t, **values))
        svc.score_all()
        t += timedelta(seconds=5)
    return t


def test_system_degradation_replaces_per_camera_alerts(service) -> None:
    from sqlalchemy import text

    svc, sent, engine = service
    t = feed_all(svc, T0, 40)  # normal: builds every camera's frames / confidence baseline
    assert sent == []
    # A traffic alert on Old Street starts just before the pipeline glitch is visible...
    incident = {"cam-old-street": {"avg_speed_kmh": 4.0, "stopped_vehicles": 4}}
    t = feed_all(svc, t, 3, per_camera=incident)
    assert [(m.event, m.camera_id) for _, m in sent] == [("start", "cam-old-street")]
    linked = sent[0][1].anomaly_id
    # ...then every camera's frames collapse (and tracking with them): a system problem.
    glitch = {"frames": 6, "avg_speed_kmh": 5.0, "stopped_vehicles": 3}
    t = feed_all(svc, t, 10, **glitch)
    system = [m for _, m in sent if m.scope == "system"]
    assert [m.event for m in system] == ["start"]
    start = system[0]
    assert start.camera_id is None and start.affected_cameras == sorted(CAMERAS)
    assert start.anomaly_type == "camera_degraded" and start.metric_name == "frames"
    assert start.details["linked_alerts"] == {"cam-old-street": str(linked)}
    # No per-camera alerts started during it, though scores were far above threshold.
    camera_starts = [m for _, m in sent if m.scope == "camera" and m.event == "start"]
    assert len(camera_starts) == 1
    assert svc.system is not None and svc.system.suppressed_checks > 0

    t = feed_all(svc, t, 8)  # pipeline recovers: the system episode ends after 3 clear checks
    ends = [m for _, m in sent if m.event == "end"]
    assert [(m.scope, m.anomaly_id) for m in ends] == [
        ("camera", linked),  # the traffic alert that started with the glitch is closed...
        ("system", start.anomaly_id),  # ...then the system episode itself
    ]
    assert ends[0].details["closed_by"] == "system_degradation"
    assert ends[0].details["system_anomaly_id"] == str(start.anomaly_id)
    assert ends[1].details["closed_alerts"] == [str(linked)]
    assert ends[1].details["suppressed_checks"] > 0
    assert svc.system is None and all(s.active is None for s in svc.cameras.values())

    feed_all(svc, t, 6)  # normal afterwards: nothing new
    assert len(sent) == len(ends) + 2
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT scope, camera_id, status, affected_cameras FROM anomalies ORDER BY scope")
        ).all()
    assert [(r.scope, r.camera_id, r.status) for r in rows] == [
        ("camera", "cam-old-street", "ended"),
        ("system", None, "ended"),
    ]
    assert rows[1].affected_cameras == sorted(CAMERAS)


def test_fewer_degraded_cameras_than_the_minimum_stay_per_camera(service) -> None:
    svc, sent, _ = service
    t = feed_all(svc, T0, 40)
    glitch = {"frames": 6}
    feed_all(svc, t, 6, per_camera={c: glitch for c in CAMERAS[:2]})  # 2 < 3 cameras
    assert svc.system is None and not [m for _, m in sent if m.scope == "system"]
    starts = {m.camera_id for _, m in sent if m.event == "start"}
    assert starts == set(CAMERAS[:2])  # handled as ordinary per-camera alerts


def test_anomaly_scope_validation() -> None:
    common = {
        "anomaly_type": "camera_degraded",
        "metric_name": "frames",
        "observed_value": 0.5,
        "score": 4.0,
        "threshold": 3.0,
        "model_name": "system-health-rules",
        "model_version": "1",
        "window_start": T0,
        "window_end": T0 + timedelta(seconds=5),
    }
    with pytest.raises(ValueError, match="camera_id is required"):
        Anomaly(**common)
    with pytest.raises(ValueError, match="affected_cameras"):
        Anomaly(scope="system", **common)
    assert Anomaly(scope="system", affected_cameras=["cam-a"], **common).camera_id is None
