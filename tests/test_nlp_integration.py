"""Integration test: TextReports published to reports.text come back as NLPResults on
nlp.results with the right camera, and are stored in reports / nlp_results.

Runs an in-process NlpService (keyword fallback classifier, no NER download) on its own
consumer group against a scratch database. The live nlp_service also sees these reports;
the online evaluation ignores them (they have no simulated_reports ground truth)."""

import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from conftest import SMART_APP_CONTROL_HINT

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "nlp_service")]

pytest.importorskip("confluent_kafka", reason=SMART_APP_CONTROL_HINT, exc_type=ImportError)
pytest.importorskip("rapidfuzz")
pytest.importorskip("mlflow")
from alembic import command  # noqa: E402
from confluent_kafka import Consumer  # noqa: E402
from sqlalchemy import text  # noqa: E402

from kafka_helpers import end_offsets, start_group_at_end, wait_for_assignment  # noqa: E402
from ml.nlp.rules import KeywordIncidentClassifier  # noqa: E402
from ml.registry import ModelManager  # noqa: E402
from nlp_service.config import NlpConfig  # noqa: E402
from nlp_service.service import NlpService  # noqa: E402
from traffic_common.db import create_db_engine  # noqa: E402
from traffic_common.db.migrate import alembic_config  # noqa: E402
from traffic_common.db.seed import seed_cameras  # noqa: E402
from traffic_common.messaging import TypedConsumer, TypedProducer  # noqa: E402
from traffic_common.schemas import NLPResult, TextReport  # noqa: E402
from traffic_common.topics import Topic  # noqa: E402

REPORTS = [
    (
        "police",
        "Collision on Piccadilly, two lanes closed, officers on scene",
        "cam-hyde-park-corner",
    ),
    ("citizen", "massive queue at Trafalger Sqaure since 8am, buses stuck", "cam-trafalgar-square"),
    ("social_media", "flooding again 🌧️ #ElephantAndCastle #LondonTraffic", "cam-elephant-castle"),
    ("operator", "Broken-down van on City Rd near Old St rbt, recovery en route", "cam-old-street"),
    ("citizen", "running late, nothing to do with traffic", None),
]


class NoRegistry:
    def get_registered_model(self, name):
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

        raise MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)


def test_reports_produce_nlp_results(kafka_bootstrap: str, scratch_database: str) -> None:
    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    seed_cameras(engine)
    models = ModelManager(
        "http://unused",
        {"classifier": "incident-text-classifier"},
        client=NoRegistry(),
        fallbacks={"classifier": KeywordIncidentClassifier},
    )
    models.refresh()
    producer = TypedProducer(kafka_bootstrap, client_id="test-nlp", register_shutdown=False)
    service = NlpService(NlpConfig(ner_enabled=False), engine, models, producer.send)
    start = end_offsets(kafka_bootstrap, Topic.NLP_RESULTS)
    group = f"test-nlp-{uuid.uuid4().hex[:8]}"
    # Start the group at the current end of reports.text (no replay, no assignment race).
    start_group_at_end(kafka_bootstrap, group, [Topic.REPORTS_TEXT])
    stop = threading.Event()
    consumer = TypedConsumer(
        [Topic.REPORTS_TEXT],
        group,
        None,
        kafka_bootstrap,
        batch_handler=service.handle_batch,
        max_batch=16,
        stop_event=stop,
        poll_timeout=0.2,
        auto_offset_reset="earliest",  # only applies if the commit above were missing
    )
    thread = threading.Thread(target=consumer.run, daemon=True)
    thread.start()
    try:
        wait_for_assignment(consumer, kafka_bootstrap)
        sent = [TextReport(source=s, text=t) for s, t, _ in REPORTS]
        with TypedProducer(kafka_bootstrap, register_shutdown=False) as p:
            for report in sent:
                p.send(Topic.REPORTS_TEXT, report)
        expected = {r.report_id: cam for r, (_, _, cam) in zip(sent, REPORTS, strict=True)}

        reader = Consumer(
            {"bootstrap.servers": kafka_bootstrap, "group.id": f"reader-{uuid.uuid4().hex}"}
        )
        reader.assign(start)
        results: dict = {}
        deadline = time.monotonic() + 60
        while len(results) < len(sent) and time.monotonic() < deadline:
            producer.flush(1)
            m = reader.poll(0.5)
            if m is None or m.error():
                continue
            result = NLPResult.model_validate_json(m.value())
            if result.report_id in expected:
                results[result.report_id] = result
        reader.close()
    finally:
        stop.set()
        thread.join(timeout=15)

    assert set(results) == set(expected), f"got {len(results)} of {len(expected)} results"
    for report_id, camera in expected.items():
        assert results[report_id].camera_id == camera, results[report_id]
    by_text = {r.text: results[r.report_id] for r in sent}
    assert by_text[REPORTS[0][1]].incident_type == "accident"
    assert by_text[REPORTS[4][1]].incident_type == "none"
    assert all(r.model_name == "incident-text-rules" for r in results.values())
    with engine.connect() as conn:
        stored = conn.execute(
            text("SELECT report_id, camera_id FROM nlp_results WHERE report_id = ANY(:ids)"),
            {"ids": list(expected)},
        ).all()
        n_reports = conn.scalar(
            text("SELECT count(*) FROM reports WHERE id = ANY(:ids)"), {"ids": list(expected)}
        )
    assert {r.report_id: r.camera_id for r in stored} == expected
    assert n_reports == len(sent)
    engine.dispose()
