"""Integration test: an Incident published to `incidents` produces a Summary on `summaries`,
stored in a scratch database with its context.

Runs an in-process SummarizerService (template configuration: no model download) on its own
consumer group, for a test camera that only exists in the scratch database."""

import json
import sys
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from conftest import SMART_APP_CONTROL_HINT

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "summarizer")]

pytest.importorskip("confluent_kafka", reason=SMART_APP_CONTROL_HINT, exc_type=ImportError)
pytest.importorskip("rapidfuzz")
pytest.importorskip("mlflow")
from alembic import command  # noqa: E402
from confluent_kafka import Consumer  # noqa: E402
from sqlalchemy import insert, text  # noqa: E402

from kafka_helpers import end_offsets, start_group_at_end, wait_for_assignment  # noqa: E402
from ml.registry import ModelManager  # noqa: E402
from ml.summarize.pyfunc import IncidentSummarizerModel  # noqa: E402
from ml.summarize.summarizer import SummarizerConfig  # noqa: E402
from summarizer_service.config import SummarizerServiceConfig  # noqa: E402
from summarizer_service.service import SummarizerService  # noqa: E402
from traffic_common.db import create_db_engine  # noqa: E402
from traffic_common.db.migrate import alembic_config  # noqa: E402
from traffic_common.db.models import Camera  # noqa: E402
from traffic_common.messaging import TypedConsumer, TypedProducer  # noqa: E402
from traffic_common.schemas import Incident, Summary  # noqa: E402
from traffic_common.topics import Topic  # noqa: E402


class NoRegistry:
    def get_registered_model(self, name):
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

        raise MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)


def template_model() -> IncidentSummarizerModel:
    model = IncidentSummarizerModel().load_config(SummarizerConfig(generator="template").to_dict())
    model.name, model.version = "incident-summary-template", "template-1"
    return model


def test_incident_produces_summary(kafka_bootstrap: str, scratch_database: str) -> None:
    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    camera = f"test-cam-{uuid.uuid4().hex[:8]}"
    incident_id = uuid.uuid4()
    start = datetime.now(UTC) - timedelta(minutes=4)
    with engine.begin() as conn:
        conn.execute(
            insert(Camera).values(
                id=camera,
                name="Test Junction - North approach",
                location_name="Test Junction, London",
                latitude=51.5,
                longitude=-0.1,
                stream_url="file:///dev/null",
            )
        )
        conn.execute(
            text(
                "INSERT INTO incidents (id, camera_id, status, incident_type, version, severity, "
                "fused_confidence, start_time, suspected_at, title, details) VALUES (:id, :c, "
                "'confirmed', 'breakdown', 2, 'high', 0.8, :t, :t, 'x', CAST(:d AS jsonb))"
            ),
            {
                "id": incident_id,
                "c": camera,
                "t": start,
                "d": json.dumps({"state": {"reports": {}, "baseline": None}}),
            },
        )
    models = ModelManager(
        "http://unused",
        {"summarizer": "incident-summarizer"},
        client=NoRegistry(),
        fallbacks={"summarizer": template_model},
    )
    models.refresh()
    producer = TypedProducer(kafka_bootstrap, client_id="test-summarizer", register_shutdown=False)
    service = SummarizerService(
        SummarizerServiceConfig(debounce_seconds=0.5), engine, models, producer.send
    )
    group = f"test-summarizer-{uuid.uuid4().hex[:8]}"
    start_group_at_end(kafka_bootstrap, group, [Topic.INCIDENTS])
    start_summaries = end_offsets(kafka_bootstrap, Topic.SUMMARIES)
    stop = threading.Event()
    consumer = TypedConsumer(
        [Topic.INCIDENTS],
        group,
        service.handle,
        kafka_bootstrap,
        stop_event=stop,
        poll_timeout=0.2,
        auto_offset_reset="earliest",
    )

    def worker() -> None:
        while not stop.wait(0.2):
            service.tick()

    threads = [
        threading.Thread(target=consumer.run, daemon=True),
        threading.Thread(target=worker, daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        wait_for_assignment(consumer, kafka_bootstrap)
        message = Incident(
            incident_id=incident_id,
            camera_id=camera,
            version=2,
            status="confirmed",
            incident_type="breakdown",
            severity="high",
            fused_confidence=0.8,
            start_time=start,
            title="Breakdown at Test Junction",
        )
        with TypedProducer(kafka_bootstrap, register_shutdown=False) as p:
            p.send(Topic.INCIDENTS, message)
        reader = Consumer(
            {"bootstrap.servers": kafka_bootstrap, "group.id": f"reader-{uuid.uuid4().hex}"}
        )
        reader.assign(start_summaries)
        found = None
        deadline = time.monotonic() + 60
        while found is None and time.monotonic() < deadline:
            producer.flush(1)
            m = reader.poll(0.5)
            if m is None or m.error():
                continue
            summary = Summary.model_validate_json(m.value())
            if summary.incident_id == incident_id:
                found = summary
        reader.close()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=15)

    assert found is not None, "no summary published"
    assert found.incident_version == 2 and found.generation == "template"
    assert found.headline == "High broken-down vehicle at Test Junction, confirmed"
    assert found.text.startswith("Confirmed broken-down vehicle at Test Junction - North approach")
    assert "Recommended action: divert traffic" in found.text
    assert found.faithfulness["fallback_reason"] == "template configuration"
    assert found.faithfulness["triggers"] == ["created"]
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT incident_version, generation, headline, context FROM summaries")
        ).one()
    assert (row.incident_version, row.generation) == (2, "template")
    assert row.context["incident_type"] == "breakdown" and row.headline == found.headline
    engine.dispose()
