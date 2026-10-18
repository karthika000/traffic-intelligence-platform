"""Drift endpoints against a scratch database: latest status per model (with features, data
quality, retraining recommendation and last promotion decision), history, alerts, the HTML
report, and switching drift mode on and off."""

import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "services" / "api")]

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
if sys.platform == "win32":  # async psycopg needs a selector event loop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from api_service.app import create_app  # noqa: E402
from api_service.config import ApiConfig  # noqa: E402
from traffic_common.schemas import DriftAlert  # noqa: E402

NOW = datetime.now(UTC).replace(microsecond=0)
OLD_CHECK, CHECK, TEXT_CHECK = uuid.UUID(int=1), uuid.UUID(int=2), uuid.UUID(int=3)
FEATURES = [
    {"feature": "frame_brightness", "kind": "num", "drift_type": "data", "method": "wasserstein",
     "score": 2.4, "threshold": 0.1, "drifted": True, "reference_rows": 800, "current_rows": 400,
     "reference_summary": {"mean": 112.0}, "current_summary": {"mean": 38.0}, "note": None},
    {"feature": "object_class", "kind": "cat", "drift_type": "prediction",
     "method": "jensenshannon", "score": 0.04, "threshold": 0.1, "drifted": False,
     "reference_rows": 9000, "current_rows": 3000, "reference_summary": {},
     "current_summary": {}, "note": None},
]  # fmt: skip


class FakeProducer:
    def send(self, topic, message):
        pass

    def flush(self, timeout=None) -> int:
        return 0


def _check(conn, check_id, model, status, share, at, features=(), alert=None, report_url=None):
    for f in features:
        conn.execute(
            text(
                "INSERT INTO drift_reports (check_id, model_name, model_version, drift_type, "
                "metric, feature, drift_score, threshold, is_drift, window_start, window_end, "
                "created_at) VALUES (:c, :m, '1', :t, :me, :f, :s, :th, :d, :ws, :we, :at)"
            ),
            {"c": check_id, "m": model, "t": f["drift_type"], "me": f["method"],
             "f": f["feature"], "s": f["score"], "th": f["threshold"], "d": f["drifted"],
             "ws": at - timedelta(minutes=15), "we": at, "at": at},
        )  # fmt: skip
    report = {
        "current_rows": 400,
        "reference_rows": 9800,
        "features": list(features),
        "drifted_features": [f["feature"] for f in features if f["drifted"]],
        "quality": [
            {"kind": "stale_camera", "target": "cam-old-street", "value": 300.0, "limit": 120.0,
             "message": "cam-old-street: no data for 300s"}
        ] if status != "ok" else [],
        "report_url": report_url,
        "mlflow": {"run_id": "run-x", "experiment_id": "9"},
        "alert": alert,
        "reference": {"source": "test"},
    }  # fmt: skip
    conn.execute(
        text(
            "INSERT INTO drift_reports (check_id, status, model_name, model_version, drift_type, "
            "metric, feature, drift_score, threshold, is_drift, window_start, window_end, "
            "report, created_at) VALUES (:c, :st, :m, '1', 'data', 'drift_share', NULL, :s, "
            "0.3, :d, :ws, :we, CAST(:r AS jsonb), :at)"
        ),
        {"c": check_id, "st": status, "m": model, "s": share, "d": status == "drifted",
         "ws": at - timedelta(minutes=15), "we": at, "r": json.dumps(report), "at": at},
    )  # fmt: skip


def seed(engine) -> DriftAlert:
    alert = DriftAlert(
        window_start=NOW - timedelta(minutes=15),
        window_end=NOW,
        model_name="vehicle-detector",
        model_version="1",
        drift_type="data",
        metric="drift_share",
        drift_score=0.5,
        threshold=0.3,
        details={"drifted_features": [{"feature": "frame_brightness", "score": 2.4}]},
    )
    with engine.begin() as conn:
        _check(conn, OLD_CHECK, "vehicle-detector", "ok", 0.0, NOW - timedelta(minutes=30))
        _check(conn, CHECK, "vehicle-detector", "drifted", 0.5, NOW, FEATURES,
               alert.model_dump(mode="json"), f"/api/v1/drift/reports/{CHECK}")  # fmt: skip
        _check(conn, TEXT_CHECK, "incident-text-classifier", "ok", 0.0, NOW - timedelta(hours=30))
        conn.execute(
            text(
                "INSERT INTO retraining_recommendations (model_name, model_version, reason, "
                "drifted_features, first_alert_at, last_alert_at) VALUES ('vehicle-detector', "
                "'1', 'drift alert', '[{\"feature\": \"frame_brightness\"}]', :t, :t)"
            ),
            {"t": NOW},
        )
        promotion = {
            "decision": "rejected",
            "promoted": False,
            "candidate_version": "2",
            "champion_version": "1",
            "reasons": ["regresses on test_severity_macro_f1"],
            "comparison": [{"metric": "heldout_mean_macro_f1", "ok": True}],
            "decided_at": NOW.isoformat(),
            "gate_run_id": "gate-1",
        }
        conn.execute(
            text(
                "INSERT INTO model_versions (model_name, version, stage, promotion) VALUES "
                "('incident-text-classifier', '2', 'archived', CAST(:p AS jsonb))"
            ),
            {"p": json.dumps(promotion)},
        )
    return alert


@pytest.fixture(scope="module")
def client(scratch_database_module: str, tmp_path_factory):
    from alembic import command

    from traffic_common.db import create_db_engine
    from traffic_common.db.migrate import alembic_config

    command.upgrade(alembic_config(scratch_database_module), "head")
    engine = create_db_engine(scratch_database_module)
    alert = seed(engine)
    reports = tmp_path_factory.mktemp("drift_reports")
    (reports / "vehicle-detector").mkdir()
    (reports / "vehicle-detector" / f"{CHECK}.html").write_text("<html>evidently</html>")
    app = create_app(
        ApiConfig(drift_reports_dir=reports),
        scratch_database_module,
        "unused:9092",
        "http://mlflow.test",
        producer=FakeProducer(),
        live_updates=False,
    )
    with TestClient(app) as c:
        c.engine, c.alert = engine, alert
        yield c
    engine.dispose()


def test_status_per_model(client) -> None:
    body = client.get("/api/v1/drift/status").json()
    by = {s["model_name"]: s for s in body}
    assert list(by)[:4] == [
        "vehicle-detector",
        "congestion-classifier",
        "traffic-anomaly-detector",
        "incident-text-classifier",
    ]
    det = by["vehicle-detector"]
    assert det["status"] == "drifted" and det["drift_share"] == 0.5  # the latest check
    assert det["check_id"] == str(CHECK) and det["share_threshold"] == 0.3
    assert det["drifted_features"] == ["frame_brightness"]
    assert det["features"][0]["current_summary"] == {"mean": 38.0}
    assert det["quality"][0]["kind"] == "stale_camera"
    assert det["report_url"] == f"/api/v1/drift/reports/{CHECK}"
    assert det["mlflow_run_id"] == "run-x"
    assert det["retraining"]["status"] == "recommended" and det["retraining"]["alert_count"] == 1
    assert by["congestion-classifier"]["status"] == "unknown"  # never checked
    text_model = by["incident-text-classifier"]
    assert text_model["status"] == "ok"
    assert text_model["last_promotion"]["decision"] == "rejected"
    assert text_model["last_promotion"]["reasons"] == ["regresses on test_severity_macro_f1"]


def test_history(client) -> None:
    points = client.get("/api/v1/drift/history", params={"hours": 2}).json()
    assert [(p["model_name"], p["status"]) for p in points] == [
        ("vehicle-detector", "ok"),
        ("vehicle-detector", "drifted"),
    ]
    assert points[1]["drifted_features"] == ["frame_brightness"]
    all_day = client.get("/api/v1/drift/history", params={"hours": 48}).json()
    assert len(all_day) == 3
    only = client.get(
        "/api/v1/drift/history", params={"hours": 48, "model_name": "incident-text-classifier"}
    ).json()
    assert [p["check_id"] for p in only] == [str(TEXT_CHECK)]
    assert client.get("/api/v1/drift/history", params={"hours": 0}).status_code == 422


def test_alerts(client) -> None:
    alerts = client.get("/api/v1/drift/alerts").json()
    assert len(alerts) == 1
    alert = DriftAlert.model_validate(alerts[0])
    assert alert == client.alert


def test_html_report(client) -> None:
    r = client.get(f"/api/v1/drift/reports/{CHECK}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "evidently" in r.text
    gone = client.get(f"/api/v1/drift/reports/{OLD_CHECK}")
    assert gone.status_code == 404 and "MLflow run run-x" in gone.json()["error"]["message"]
    assert client.get(f"/api/v1/drift/reports/{uuid.uuid4()}").status_code == 404
    assert client.get("/api/v1/drift/reports/not-a-uuid").status_code == 422


def test_drift_mode_toggle(client) -> None:
    assert client.get("/api/v1/drift/mode").json()["active"] is False
    on = client.post(
        "/api/v1/drift/mode", json={"enabled": True, "preset": "fog", "duration_minutes": 20}
    )
    assert on.status_code == 200
    body = on.json()
    assert body["active"] and body["settings"]["preset"] == "fog" and body["updated_by"] == "api"
    assert body["visual_params"]["haze"] > 0
    expires = datetime.fromisoformat(body["expires_at"])
    assert timedelta(minutes=19) < expires - datetime.now(UTC) <= timedelta(minutes=20)
    assert client.get("/api/v1/drift/mode").json()["active"] is True

    off = client.post("/api/v1/drift/mode", json={"enabled": False}).json()
    assert off["active"] is False and off["expires_at"] is None
    assert client.get("/api/v1/drift/mode").json()["active"] is False
    with client.engine.connect() as conn:
        events = conn.execute(text("SELECT enabled FROM drift_mode_events ORDER BY id")).all()
    assert [e.enabled for e in events] == [True, False]


def test_drift_mode_validation(client) -> None:
    bad = client.post("/api/v1/drift/mode", json={"enabled": True, "brightness": 3})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "validation_error"
    assert client.post("/api/v1/drift/mode", json={"enabled": True, "x": 1}).status_code == 422


def test_dismiss_retraining_recommendation(client) -> None:
    r = client.post(
        "/api/v1/drift/recommendations/vehicle-detector/dismiss",
        json={"reason": "alerts from the drift demo", "dismissed_by": "test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "dismissed" and body["resolved_at"]
    assert body["resolution"] == {
        "decision": "dismissed",
        "reason": "alerts from the drift demo",
        "by": "test",
    }
    status = {s["model_name"]: s for s in client.get("/api/v1/drift/status").json()}
    assert status["vehicle-detector"]["retraining"]["status"] == "dismissed"
    # Nothing open any more, or no recommendation at all: 404.
    again = client.post(
        "/api/v1/drift/recommendations/vehicle-detector/dismiss", json={"reason": "again"}
    )
    assert again.status_code == 404
    assert (
        client.post(
            "/api/v1/drift/recommendations/congestion-classifier/dismiss", json={"reason": "x y z"}
        ).status_code
        == 404
    )
    bad = client.post("/api/v1/drift/recommendations/vehicle-detector/dismiss", json={"reason": ""})
    assert bad.status_code == 422
