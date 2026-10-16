"""Migration tests against a throwaway database on the running Postgres server."""

import uuid
from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from traffic_common.db import Base, create_db_engine, models
from traffic_common.db.migrate import alembic_config
from traffic_common.db.seed import DEMO_CAMERAS, seed_cameras

EXPECTED_TABLES = {
    "cameras",
    "traffic_metrics",
    "detection_aggregates",
    "reports",
    "nlp_results",
    "anomalies",
    "incidents",
    "incident_evidence",
    "summaries",
    "model_versions",
    "drift_reports",
    "scenario_runs",
    "simulated_reports",
    "sensor_readings",
    "camera_congestion",
}


def test_single_head_revision() -> None:
    heads = ScriptDirectory.from_config(alembic_config("postgresql://x@localhost/x")).get_heads()
    assert heads == ["0009"]


def test_upgrade_and_downgrade(scratch_database: str) -> None:
    cfg = alembic_config(scratch_database)
    engine = create_db_engine(scratch_database)
    try:
        command.upgrade(cfg, "head")
        insp = inspect(engine)
        assert set(insp.get_table_names()) == EXPECTED_TABLES | {"alembic_version"}

        with engine.connect() as conn:
            # The migration produces exactly what the models describe.
            diff = compare_metadata(
                MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata
            )
            assert diff == []
            assert conn.scalar(text("SELECT version_num FROM alembic_version")) == "0009"

        indexes = {i["name"] for i in insp.get_indexes("incidents")}
        assert {"ix_incidents_camera_id_start_time", "ix_incidents_start_time"} <= indexes
        checks = {c["name"] for c in insp.get_check_constraints("incidents")}
        assert {
            "ck_incidents_incident_status",
            "ck_incidents_severity",
            "ck_incidents_fused_confidence_range",
        } <= checks

        command.downgrade(cfg, "base")
        assert set(inspect(engine).get_table_names()) == {"alembic_version"}

        command.upgrade(cfg, "head")  # and back up again cleanly
        assert EXPECTED_TABLES <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_seed_is_idempotent(scratch_database: str) -> None:
    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    try:
        assert seed_cameras(engine) == 4
        with engine.begin() as conn:  # what the zone and video scripts would write
            conn.execute(
                text(
                    "UPDATE cameras SET zone_config = '{\"pixels_per_metre\": 40}',"
                    " stream_url = '/data/videos/custom.webm' WHERE id = 'cam-old-street'"
                )
            )
        assert seed_cameras(engine) == 4  # re-seed
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT id, latitude, longitude, zone_config, stream_url FROM cameras")
            ).all()
        assert {r.id for r in rows} == {c["id"] for c in DEMO_CAMERAS}
        for r in rows:
            assert 51.4 < r.latitude < 51.6 and -0.2 < r.longitude < 0.0  # central London
        old_street = next(r for r in rows if r.id == "cam-old-street")
        assert old_street.zone_config == {"pixels_per_metre": 40}  # re-seed kept calibration
        assert old_street.stream_url == "/data/videos/custom.webm"
    finally:
        engine.dispose()


def test_constraints_enforced(scratch_database: str) -> None:
    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    seed_cameras(engine)
    now = datetime.now(UTC)
    incident_id, report_id = uuid.uuid4(), uuid.uuid4()
    try:
        with engine.begin() as conn:
            conn.execute(
                models.Incident.__table__.insert().values(
                    id=incident_id,
                    camera_id="cam-old-street",
                    severity="high",
                    fused_confidence=0.8,
                    start_time=now,
                    title="Collision",
                )
            )
            conn.execute(
                models.Report.__table__.insert().values(
                    id=report_id, source="operator", text="Collision reported", reported_at=now
                )
            )
            conn.execute(
                models.IncidentEvidence.__table__.insert().values(
                    incident_id=incident_id, evidence_type="report", report_id=report_id, weight=0.6
                )
            )
            status = conn.scalar(text("SELECT status FROM incidents"))
            assert status == "suspected"  # server default

        bad_rows = [
            # evidence type doesn't match the populated reference
            (
                models.IncidentEvidence,
                dict(incident_id=incident_id, evidence_type="anomaly", report_id=report_id),
            ),
            # resolved without an end time
            (
                models.Incident,
                dict(
                    id=uuid.uuid4(),
                    camera_id="cam-old-street",
                    status="resolved",
                    severity="low",
                    fused_confidence=0.5,
                    start_time=now,
                    title="x",
                ),
            ),
            # unknown camera
            (
                models.Incident,
                dict(
                    id=uuid.uuid4(),
                    camera_id="cam-nowhere",
                    severity="low",
                    fused_confidence=0.5,
                    start_time=now,
                    title="x",
                ),
            ),
        ]
        for model, values in bad_rows:
            with pytest.raises(IntegrityError), engine.begin() as conn:
                conn.execute(model.__table__.insert().values(**values))

        with (
            pytest.raises(IntegrityError),
            engine.begin() as conn,
        ):  # status CHECK, bypassing the ORM enum
            conn.execute(text("UPDATE incidents SET status = 'imagined'"))
    finally:
        engine.dispose()
