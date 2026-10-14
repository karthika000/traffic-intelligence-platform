"""Integration test: trigger a scenario through simulator.control and check that its
tagged messages arrive on reports.text / events.raw and its ground truth lands in Postgres.

Uses a scratch database with a `test-cam-...` camera and its own consumer group. The live
event_simulator (group event-simulator) also sees the command but rejects it, because the
camera doesn't exist in the real database. The emitted messages stay on the real topics
until Kafka retention removes them; they are tagged with the test scenario_id and camera.
"""

import random
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import insert, select

from conftest import SMART_APP_CONTROL_HINT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "event_simulator"))

pytest.importorskip("confluent_kafka", reason=SMART_APP_CONTROL_HINT, exc_type=ImportError)
pytest.importorskip("yaml")
from alembic import command  # noqa: E402
from confluent_kafka import Consumer, TopicPartition  # noqa: E402

from event_simulator.config import SimulatorConfig  # noqa: E402
from event_simulator.emitter import Emitter  # noqa: E402
from event_simulator.groundtruth import GroundTruthStore  # noqa: E402
from event_simulator.simulator import Simulator  # noqa: E402
from event_simulator.trigger import send_command  # noqa: E402
from traffic_common.db import create_db_engine  # noqa: E402
from traffic_common.db.migrate import alembic_config  # noqa: E402
from traffic_common.db.models import Camera, ScenarioRun, SimulatedReport  # noqa: E402
from traffic_common.messaging import TypedConsumer, TypedProducer  # noqa: E402
from traffic_common.schemas import RawEvent, SimulatorCommand, TextReport  # noqa: E402
from traffic_common.topics import Topic  # noqa: E402

TIME_SCALE = 240  # road_works: 20 min -> 5 s


def end_offsets(bootstrap: str, topic: str) -> list[TopicPartition]:
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": f"probe-{uuid.uuid4().hex}"})
    try:
        parts = c.list_topics(topic, timeout=10).topics[topic].partitions
        return [
            TopicPartition(
                topic, p, c.get_watermark_offsets(TopicPartition(topic, p), timeout=10)[1]
            )
            for p in parts
        ]
    finally:
        c.close()


def read_tagged(
    bootstrap: str,
    start: list[TopicPartition],
    scenario_id: uuid.UUID,
    schema,
    expected: int,
    timeout: float = 20,
) -> list:
    c = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"reader-{uuid.uuid4().hex}",
            "enable.auto.commit": False,
        }
    )
    c.assign(start)
    found, deadline = [], time.monotonic() + timeout
    try:
        while len(found) < expected and time.monotonic() < deadline:
            m = c.poll(0.5)
            if m is None or m.error():
                continue
            msg = schema.model_validate_json(m.value())
            if msg.scenario_id == scenario_id:
                found.append(msg)
        return found
    finally:
        c.close()


def test_triggered_scenario_reaches_kafka_and_postgres(
    kafka_bootstrap: str, scratch_database: str
) -> None:
    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    camera_id = f"test-cam-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        conn.execute(
            insert(Camera).values(
                id=camera_id,
                name="Test Junction - North Approach",
                location_name="Test Junction, X1",
                latitude=51.5,
                longitude=-0.1,
                stream_url="/data/videos/none.webm",
            )
        )

    starts = {t: end_offsets(kafka_bootstrap, t) for t in (Topic.REPORTS_TEXT, Topic.EVENTS_RAW)}
    stop = threading.Event()
    store = GroundTruthStore(engine)
    with TypedProducer(kafka_bootstrap, register_shutdown=False) as producer:
        sim = Simulator(
            SimulatorConfig(), engine, Emitter(producer.send, store), store, stop, random.Random(7)
        )
        sim.refresh_cameras()
        assert camera_id in sim.cameras
        control = TypedConsumer(
            [Topic.SIMULATOR_CONTROL],
            f"test-sim-{uuid.uuid4().hex[:8]}",
            sim.handle_command,
            kafka_bootstrap,
            stop_event=stop,
            auto_offset_reset="latest",
            poll_timeout=0.2,
        )
        control_thread = threading.Thread(target=control.run, daemon=True)
        control_thread.start()
        try:
            cmd = SimulatorCommand(
                action="start_scenario",
                scenario="road_works",
                camera_id=camera_id,
                time_scale=TIME_SCALE,
            )
            # The group joins asynchronously and starts at "latest": resend until it is picked up.
            deadline = time.monotonic() + 30
            while camera_id not in sim.runs and time.monotonic() < deadline:
                send_command(cmd, kafka_bootstrap)
                time.sleep(2)
            runner = sim.runs[camera_id]
            assert runner.run_id == cmd.scenario_id
            runner.join(timeout=30)
            assert runner.status.value == "completed"
        finally:
            stop.set()
            control_thread.join(timeout=15)

    reports = read_tagged(
        kafka_bootstrap,
        starts[Topic.REPORTS_TEXT],
        cmd.scenario_id,
        TextReport,
        runner.reports_emitted,
    )
    events = read_tagged(
        kafka_bootstrap, starts[Topic.EVENTS_RAW], cmd.scenario_id, RawEvent, runner.events_emitted
    )
    assert len(reports) == runner.reports_emitted >= 15
    assert len(events) == runner.events_emitted >= 60
    assert {r.camera_id for r in reports} == {e.camera_id for e in events} == {camera_id}
    assert {e.event_type for e in events} == {"weather", "loop_detector"}

    with engine.connect() as conn:
        run = conn.execute(select(ScenarioRun).where(ScenarioRun.id == cmd.scenario_id)).one()
        truth = conn.execute(
            select(SimulatedReport).where(SimulatedReport.scenario_run_id == cmd.scenario_id)
        ).all()
    engine.dispose()
    assert run.status == "completed" and run.scenario_name == "road_works"
    assert run.reports_emitted == len(reports) and run.events_emitted == len(events)
    assert [p["phase"] for p in run.timeline] == [
        "announced",
        "lane_closed",
        "overrun",
        "completed",
    ]
    assert {t.id for t in truth} == {r.report_id for r in reports}
    by_id = {r.report_id: r for r in reports}
    for t in truth:
        assert by_id[t.id].text == t.text
        assert t.incident_type in ("road_works", "congestion")
        assert t.location_mention is None or t.location_mention in t.text
    assert {t.phase for t in truth} <= {"announced", "lane_closed", "overrun", "completed"}
    assert any(t.stage == "cleared" for t in truth)
