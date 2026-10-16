"""Integration test: an anomaly plus matching reports published to Kafka come out of fusion as
one confirmed incident on `incidents`, stored in a scratch database.

Runs an in-process FusionService on its own consumer group, for a test camera that only
exists in the scratch database (the live fusion_service ignores cameras it doesn't know)."""

import sys
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from conftest import SMART_APP_CONTROL_HINT

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "fusion_service")]

pytest.importorskip("confluent_kafka", reason=SMART_APP_CONTROL_HINT, exc_type=ImportError)
pytest.importorskip("rapidfuzz")
from alembic import command  # noqa: E402
from confluent_kafka import Consumer  # noqa: E402
from sqlalchemy import insert, text  # noqa: E402

from fusion_service.config import FusionConfig  # noqa: E402
from fusion_service.service import FusionService  # noqa: E402
from test_anomaly_integration import end_offsets  # noqa: E402
from traffic_common.db import create_db_engine  # noqa: E402
from traffic_common.db.migrate import alembic_config  # noqa: E402
from traffic_common.db.models import Camera  # noqa: E402
from traffic_common.messaging import TypedConsumer, TypedProducer  # noqa: E402
from traffic_common.schemas import Anomaly, Entity, Incident, NLPResult  # noqa: E402
from traffic_common.topics import Topic  # noqa: E402


def test_anomaly_and_reports_make_one_confirmed_incident(
    kafka_bootstrap: str, scratch_database: str
) -> None:
    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    camera = f"test-cam-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        conn.execute(
            insert(Camera).values(
                id=camera,
                name="Test junction",
                location_name="Test junction, London",
                latitude=51.5,
                longitude=-0.1,
                stream_url="file:///dev/null",
            )
        )
    producer = TypedProducer(kafka_bootstrap, client_id="test-fusion", register_shutdown=False)
    service = FusionService(FusionConfig(), engine, producer.send)
    start = end_offsets(kafka_bootstrap, Topic.INCIDENTS)
    stop = threading.Event()
    group = f"test-fusion-{uuid.uuid4().hex[:8]}"
    # Start the group at the current end of both topics: no replay of their history, and no
    # race with the first partition assignment.
    seed = Consumer({"bootstrap.servers": kafka_bootstrap, "group.id": group})
    seed.commit(
        offsets=end_offsets(kafka_bootstrap, Topic.ANOMALIES)
        + end_offsets(kafka_bootstrap, Topic.NLP_RESULTS),
        asynchronous=False,
    )
    seed.close()
    consumer = TypedConsumer(
        [Topic.ANOMALIES, Topic.NLP_RESULTS],
        group,
        service.handle,
        kafka_bootstrap,
        stop_event=stop,
        poll_timeout=0.2,
        auto_offset_reset="earliest",  # only applies if the commit above were missing
    )
    thread = threading.Thread(target=consumer.run, daemon=True)
    thread.start()
    try:
        now = datetime.now(UTC)
        anomaly = Anomaly(
            camera_id=camera,
            event="start",
            severity="high",
            anomaly_type="speed_drop",
            hints=["speed_drop", "stopped_vehicles"],
            metric_name="loop_speed_kmh_rel",
            observed_value=0.2,
            score=3.5,
            threshold=1.0,
            model_name="m",
            model_version="1",
            started_at=now - timedelta(seconds=10),
            window_start=now - timedelta(seconds=60),
            window_end=now,
            congestion_level="heavy",
        )
        reports = [
            NLPResult(
                report_id=uuid.uuid4(),
                source=source,
                incident_type="accident",
                incident_confidence=0.97,
                severity="high",
                severity_confidence=0.9,
                entities=[Entity(text="Test junction", label="LOCATION", start=0, end=13)],
                camera_id=camera,
                camera_confidence=0.95,
                model_name="m",
                model_version="1",
            )
            for source in ("citizen", "social_media", "police", "operator")
        ]
        with TypedProducer(kafka_bootstrap, register_shutdown=False) as p:
            p.send(Topic.ANOMALIES, anomaly)
            for r in reports:
                p.send(Topic.NLP_RESULTS, r)

        reader = Consumer(
            {"bootstrap.servers": kafka_bootstrap, "group.id": f"reader-{uuid.uuid4().hex}"}
        )
        reader.assign(start)
        versions: list[Incident] = []
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not any(v.status == "confirmed" for v in versions):
            producer.flush(1)
            m = reader.poll(0.5)
            if m is None or m.error():
                continue
            incident = Incident.model_validate_json(m.value())
            if incident.camera_id == camera:
                versions.append(incident)
        reader.close()
        # Keep consuming until every report has been attached and persisted.
        deadline = time.monotonic() + 90  # all four reports persisted
        while True:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT status, version, incident_type, confidence_components, details "
                        "FROM incidents"
                    )
                ).one()
            if len(row.details["state"]["reports"]) == len(reports) or time.monotonic() > deadline:
                break
            time.sleep(0.5)
    finally:
        stop.set()
        thread.join(timeout=15)

    assert versions, "no incident published"
    assert len({v.incident_id for v in versions}) == 1  # one incident, several versions
    final = versions[-1]
    assert final.status == "confirmed" and final.incident_type == "accident"
    assert [v.version for v in versions] == list(range(1, len(versions) + 1))
    assert versions[0].status == "suspected"
    assert final.anomaly_ids == [anomaly.anomaly_id]
    # Confirmed once two agreeing reports joined the anomaly; later reports still attach
    # (persisted) but move the confidence too little for a new version.
    assert set(final.report_ids) <= {r.report_id for r in reports}
    assert len(final.report_ids) >= 2
    comps = final.confidence_components
    assert comps["agreement"]["score"] == 1.0 and comps["vision"]["present"]
    assert comps["text"]["n_sources"] >= 2
    assert final.severity == "high"
    assert set(row.details["state"]["reports"]) == {str(r.report_id) for r in reports}
    assert row.confidence_components["text"]["n_sources"] == 4
    assert (row.status, row.incident_type) == ("confirmed", "accident")
    assert row.version >= final.version
    assert row.confidence_components["fusion"]["fused"] >= final.fused_confidence
    engine.dispose()
