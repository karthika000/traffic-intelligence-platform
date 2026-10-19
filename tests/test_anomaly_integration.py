"""Integration test: a synthetic incident published to traffic.metrics produces an Anomaly
on `anomalies` and a row in the anomalies table.

Runs an in-process AnomalyService (rule-based models, its own consumer group, a scratch
database with a test camera). The live anomaly_service also consumes these messages but
can't store anything for the test camera, which only exists in the scratch database.
"""

import sys
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from conftest import SMART_APP_CONTROL_HINT

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "anomaly_service")]

pytest.importorskip("confluent_kafka", reason=SMART_APP_CONTROL_HINT, exc_type=ImportError)
pytest.importorskip("pandas")
pytest.importorskip("mlflow")
from alembic import command  # noqa: E402
from confluent_kafka import Consumer  # noqa: E402
from sqlalchemy import insert, text  # noqa: E402

from anomaly_service.config import AnomalyConfig  # noqa: E402
from anomaly_service.service import AnomalyService  # noqa: E402
from kafka_helpers import end_offsets, start_group_at_end, wait_for_assignment  # noqa: E402
from test_anomaly_service import FakeRegistry, manager  # noqa: E402
from traffic_common.db import create_db_engine  # noqa: E402
from traffic_common.db.migrate import alembic_config  # noqa: E402
from traffic_common.db.models import Camera  # noqa: E402
from traffic_common.messaging import TypedConsumer, TypedProducer  # noqa: E402
from traffic_common.schemas import Anomaly, TrafficMetrics  # noqa: E402
from traffic_common.topics import Topic  # noqa: E402


def test_incident_sequence_raises_anomaly(kafka_bootstrap: str, scratch_database: str) -> None:
    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    camera_id = f"test-cam-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        conn.execute(
            insert(Camera).values(
                id=camera_id,
                name="Test",
                location_name="Test",
                latitude=51.5,
                longitude=-0.1,
                stream_url="x",
            )
        )

    mm = manager(FakeRegistry(), [], [])
    mm.refresh()  # rule-based models
    stop = threading.Event()
    with TypedProducer(kafka_bootstrap, register_shutdown=False) as producer:
        # One scoring pass after the burst, so a single check above threshold must suffice.
        cfg = AnomalyConfig(warmup_seconds=0, max_metric_age_seconds=600, min_consecutive=1)
        service = AnomalyService(cfg, engine, mm, producer.send)
        group = f"test-anomaly-{uuid.uuid4().hex[:8]}"
        start_group_at_end(kafka_bootstrap, group, [Topic.TRAFFIC_METRICS])
        consumer = TypedConsumer(
            [Topic.TRAFFIC_METRICS],
            group,
            service.handle,
            kafka_bootstrap,
            stop_event=stop,
            poll_timeout=0.2,
            auto_offset_reset="earliest",  # only applies if the commit above were missing
        )
        thread = threading.Thread(target=consumer.run, daemon=True)
        thread.start()
        wait_for_assignment(consumer, kafka_bootstrap)
        start = end_offsets(kafka_bootstrap, Topic.ANOMALIES)

        # 30 normal windows, then a sudden incident: speed collapses, vehicles stop.
        now = datetime.now(UTC)
        windows = [(30.0, 0, 2.0)] * 30 + [(4.0, 5, 7.0)] * 6
        for i, (speed, stopped, density) in enumerate(windows):
            end = now - timedelta(seconds=5 * (len(windows) - i))
            producer.send(
                Topic.TRAFFIC_METRICS,
                TrafficMetrics(
                    camera_id=camera_id,
                    window_start=end - timedelta(seconds=5),
                    window_end=end,
                    timestamp=end,
                    vehicle_count=6,
                    avg_speed_kmh=speed,
                    p85_speed_kmh=speed * 1.3,
                    density=density,
                    stopped_vehicles=stopped,
                    frames=24,
                    mean_confidence=0.6,
                ),
            )
        producer.flush()

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and service.stats.metrics < len(windows):
            time.sleep(0.2)
        service.score_all()
        stop.set()
        thread.join(timeout=15)

    reader = Consumer(
        {
            "bootstrap.servers": kafka_bootstrap,
            "group.id": f"reader-{uuid.uuid4().hex}",
            "enable.auto.commit": False,
        }
    )
    reader.assign(start)
    found, deadline = [], time.monotonic() + 15
    while not found and time.monotonic() < deadline:
        m = reader.poll(0.5)
        if m is not None and not m.error() and m.key() == camera_id.encode():
            found.append(Anomaly.model_validate_json(m.value()))
    reader.close()

    assert found, "no Anomaly for the test camera on the anomalies topic"
    anomaly = found[0]
    assert anomaly.event == "start" and anomaly.score >= anomaly.threshold
    assert {"speed_drop", "stopped_vehicles"} & set(anomaly.hints)
    assert anomaly.model_name == "anomaly-rules"
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT camera_id, status FROM anomalies WHERE id = :i"), {"i": anomaly.anomaly_id}
        ).one()
    engine.dispose()
    assert (row.camera_id, row.status) == (camera_id, "active")
