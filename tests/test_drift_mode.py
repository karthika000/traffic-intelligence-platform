"""Drift mode (traffic_common.driftmode): settings and presets, switching it on and off in
Postgres (with expiry and history), the services' watcher, video_ingest's frame degradation
and event_simulator's switch to the slang report style."""

import random
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from traffic_common.driftmode import (
    VISUAL_PRESETS,
    DriftModeSettings,
    DriftModeState,
    DriftModeWatcher,
    active_periods,
    read_state,
    set_state,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "services" / "video_ingest"), str(ROOT / "services" / "event_simulator")]

NOW = datetime(2026, 10, 17, 15, 0, tzinfo=UTC)


# ----------------------------------------------------------------------------- settings, state


def test_presets_and_overrides() -> None:
    assert DriftModeSettings().visual_params() == VISUAL_PRESETS["night"]
    fog = DriftModeSettings(preset="fog", noise_std=0.0).visual_params()
    assert fog["haze"] == VISUAL_PRESETS["fog"]["haze"] and fog["noise_std"] == 0.0
    with pytest.raises(ValueError):
        DriftModeSettings(brightness=2.0)
    with pytest.raises(ValueError):
        DriftModeSettings(unknown=1)


def test_state_activity_expiry_and_parts() -> None:
    assert not DriftModeState().active(NOW)
    on = DriftModeState(enabled=True)
    assert on.active(NOW) and on.visual_active(NOW) and on.text_style(NOW) == "slang"
    expired = DriftModeState(enabled=True, expires_at=NOW - timedelta(seconds=1))
    assert not expired.active(NOW) and expired.text_style(NOW) is None
    text_only = DriftModeState(enabled=True, settings=DriftModeSettings(visual=False))
    assert not text_only.visual_active(NOW) and text_only.text_style(NOW) == "slang"


@pytest.fixture
def engine(scratch_database: str):
    from alembic import command

    from traffic_common.db import create_db_engine
    from traffic_common.db.migrate import alembic_config

    command.upgrade(alembic_config(scratch_database), "head")
    eng = create_db_engine(scratch_database)
    yield eng
    eng.dispose()


def test_switch_on_and_off_in_postgres(engine) -> None:
    assert not read_state(engine).active()  # no row yet: off
    state = set_state(engine, True, DriftModeSettings(preset="fog"), duration_minutes=30)
    assert state.active() and state.settings.preset == "fog" and state.updated_by == "cli"
    assert state.expires_at > datetime.now(UTC) + timedelta(minutes=29)
    assert read_state(engine).settings.preset == "fog"
    state = set_state(engine, False, updated_by="test")
    assert not state.active() and state.expires_at is None
    with engine.connect() as conn:
        events = conn.execute(text("SELECT enabled FROM drift_mode_events ORDER BY id")).all()
    assert [e.enabled for e in events] == [True, False]
    # The "on" period is found when it overlaps a window, and not otherwise.
    now = datetime.now(UTC)
    assert len(active_periods(engine, now - timedelta(hours=1), now + timedelta(minutes=1))) == 1
    assert active_periods(engine, now + timedelta(minutes=5), now + timedelta(hours=1)) == []


def test_running_drift_mode_period_ends_now_and_is_left_out_of_training_data(engine) -> None:
    pd = pytest.importorskip("pandas")
    from ml.data.real import drop_drift_mode

    set_state(engine, True)  # on, no expiry: the period is still running
    now = datetime.now(UTC)
    ((start, end),) = active_periods(engine, now - timedelta(hours=1), now + timedelta(hours=1))
    assert start <= now and end <= datetime.now(UTC) + timedelta(seconds=1)
    metrics = pd.DataFrame(
        {
            "camera_id": "cam-a",
            "ts": [now - timedelta(minutes=10), now],
            "vehicle_count": [5, 1],
        }
    )
    kept = drop_drift_mode(engine, metrics, now - timedelta(hours=1), now + timedelta(hours=1))
    assert kept["vehicle_count"].tolist() == [5]


def test_expired_drift_mode_reads_as_off(engine) -> None:
    set_state(engine, True, duration_minutes=1)
    with engine.begin() as conn:
        conn.execute(text("UPDATE drift_mode SET expires_at = now() - interval '1 second'"))
    state = read_state(engine)
    assert state.enabled and not state.active()


def test_watcher_follows_changes_and_keeps_state_on_errors() -> None:
    states = [DriftModeState(), DriftModeState(enabled=True), DriftModeState(enabled=True)]
    changes = []

    def reader() -> DriftModeState:
        if not states:
            raise ConnectionError("database down")
        return states.pop(0)

    watcher = DriftModeWatcher(None, threading.Event(), reader=reader, on_change=changes.append)
    assert not watcher.poll().active()
    assert watcher.poll().active()
    watcher.poll()  # unchanged: no callback
    assert watcher.poll().active()  # read error: last state kept
    assert [s.active() for s in changes] == [False, True]


# ----------------------------------------------------------------------------- video_ingest


def test_degrade_night_and_fog() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from video_ingest.degrade import degrade

    rng = np.random.default_rng(0)
    frame = rng.integers(60, 200, size=(120, 160, 3), dtype=np.uint8)
    night = degrade(frame, DriftModeSettings(preset="night").visual_params(), rng)
    assert night.shape == frame.shape and night.dtype == np.uint8
    assert night.mean() < frame.mean() * 0.45
    fog = degrade(frame, DriftModeSettings(preset="fog", noise_std=0).visual_params(), rng)
    assert fog.std() < frame.std() * 0.5  # haze and blur flatten contrast
    assert fog.mean() > frame.mean()  # towards light grey
    noisy = degrade(np.full((50, 50, 3), 100, np.uint8), {"noise_std": 20.0}, rng)
    assert 15 < noisy.astype(float).std() < 25
    assert (degrade(frame, {}, rng) == frame).all()


def test_worker_degrades_frames_only_while_drift_mode_is_on(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    from video_ingest.config import IngestConfig
    from video_ingest.sources import resolve_source
    from video_ingest.storage import FrameStore
    from video_ingest.worker import CameraWorker

    params: dict[str, float] | None = None
    published = []
    worker = CameraWorker(
        "cam-a",
        resolve_source("cam.mp4", tmp_path),
        IngestConfig(frames_dir=tmp_path),
        FrameStore(tmp_path),
        published.append,
        drift=lambda: params,
    )
    frame = np.full((90, 160, 3), 150, np.uint8)

    def brightness() -> float:
        return float(cv2.imread(published[-1].frame_path).mean())

    worker._emit(frame, 25.0)
    assert brightness() == pytest.approx(150, abs=3)
    params = DriftModeSettings(preset="night", noise_std=0).visual_params()
    worker._emit(frame, 25.0)
    assert brightness() == pytest.approx(45, abs=3)
    params = None  # switched off again
    worker._emit(frame, 25.0)
    assert brightness() == pytest.approx(150, abs=3)


# ----------------------------------------------------------------------------- event_simulator


def test_simulator_switches_report_style_with_drift_mode() -> None:
    from event_simulator import slang
    from event_simulator.config import SimulatorConfig
    from event_simulator.emitter import Emitter
    from event_simulator.locations import CameraInfo
    from event_simulator.simulator import Simulator

    sim = Simulator(
        SimulatorConfig(scenarios_dir=ROOT / "services" / "event_simulator" / "scenarios"),
        None,
        Emitter(lambda topic, message: None, None),
        None,
        threading.Event(),
        rng=random.Random(3),
    )
    camera = CameraInfo("cam-a", "Hyde Park Corner", "Hyde Park Corner, London", 51.5, -0.15)
    standard = [sim.generator.generate(camera) for _ in range(50)]
    assert all(r.text_style is None for r in standard)

    sim.drift_mode._state = DriftModeState(enabled=True)  # what the watcher reads from Postgres
    drifted = [sim.generator.generate(camera) for _ in range(200)]
    assert all(r.text_style == slang.STYLE for r in drifted)
    assert {r.style.value for r in drifted} <= {"driver", "social_media"}
    assert all(r.location_mention in r.text for r in drifted)
    assert len({r.incident_type for r in drifted}) == 8  # every label still occurs

    sim.drift_mode._state = DriftModeState(enabled=True, settings=DriftModeSettings(text=False))
    assert sim.generator.generate(camera).text_style is None
    sim.drift_mode._state = DriftModeState()
    assert sim.generator.generate(camera).text_style is None


def test_slang_phrasing_is_not_in_the_training_pools() -> None:
    from event_simulator import reports, slang

    standard = " ".join(
        p.lower()
        for pools in (reports.CORE.values(), *(d.values() for d in reports.DETAIL.values()))
        for pool in pools
        for p in pool.all()
    )
    for phrases in (*slang.CORE.values(), *slang.IMPACT.values()):
        for phrase in phrases:
            assert phrase.lower() not in standard, phrase
