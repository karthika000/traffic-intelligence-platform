"""REST API against a scratch database seeded with a small, known world: cameras, metrics,
congestion, anomalies (one system-wide), an incident with evidence and summaries, reports
(one unlocated), model versions and scenario runs. MLflow is mocked; the Kafka producer is
fake (live updates have their own test, test_api_live.py)."""

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
httpx = pytest.importorskip("httpx")
if sys.platform == "win32":  # async psycopg needs a selector event loop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from api_service.app import create_app  # noqa: E402
from api_service.config import ApiConfig  # noqa: E402
from traffic_common.schemas import SimulatorCommand  # noqa: E402

NOW = datetime.now(UTC).replace(microsecond=0)
CAM, OTHER = "cam-old-street", "cam-hyde-park-corner"
INCIDENT = uuid.UUID(int=101)
ANOMALY, SYSTEM = uuid.UUID(int=201), uuid.UUID(int=202)
REPORTS = [uuid.UUID(int=301), uuid.UUID(int=302)]
UNLOCATED = uuid.UUID(int=303)
METRIC_MESSAGE = uuid.UUID(int=401)
JPEG = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"


class FakeProducer:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, topic, message):
        self.sent.append((topic, message))

    def flush(self, timeout=None) -> int:
        return 0


def mlflow_handler(request: httpx.Request) -> httpx.Response:
    path, params = request.url.path, dict(request.url.params)
    if path.endswith("registered-models/alias"):
        if params["name"] == "incident-text-classifier":
            return httpx.Response(
                200,
                json={
                    "model_version": {
                        "version": "1",
                        "run_id": "run-1",
                        "description": "distilbert",
                        "tags": [{"key": "candidate", "value": "distilbert"}],
                    }
                },
            )
        return httpx.Response(404, json={"error_code": "RESOURCE_DOES_NOT_EXIST"})
    if path.endswith("runs/get"):
        return httpx.Response(
            200,
            json={
                "run": {
                    "info": {"run_id": "run-1", "run_name": "distilbert", "end_time": 1},
                    "data": {
                        "metrics": [{"key": "test_incident_type_macro_f1", "value": 0.956}],
                        "tags": [{"key": "mlflow.user", "value": "x"}],
                    },
                }
            },
        )
    if path.endswith("experiments/get-by-name"):
        return httpx.Response(200, json={"experiment": {"experiment_id": "5"}})
    if path.endswith("runs/search"):
        return httpx.Response(
            200,
            json={
                "runs": [
                    {
                        "info": {"run_id": "run-online", "run_name": "online-eval"},
                        "data": {
                            "metrics": [{"key": "online_all_incident_accuracy", "value": 0.98}]
                        },
                    }
                ]
            },
        )
    return httpx.Response(500)


def seed(engine) -> None:
    j = json.dumps
    state = {
        "anomalies": {str(ANOMALY): {}},
        "reports": {str(r): {} for r in REPORTS},
        "metric_ids": [str(METRIC_MESSAGE)],
    }
    history = [
        {
            "version": 1,
            "status": "suspected",
            "fused": 0.6,
            "severity": "medium",
            "incident_type": "accident",
            "reasons": ["created: traffic anomaly"],
        },
        {
            "version": 2,
            "status": "confirmed",
            "fused": 0.9,
            "severity": "high",
            "incident_type": "accident",
            "reasons": ["confirmed (both modalities agree)"],
        },
    ]
    components = {
        "vision": {"score": 0.8},
        "text": {"score": 0.9},
        "agreement": {"score": 1.0},
        "fusion": {"fused": 0.9},
    }
    stmts = [
        (
            "INSERT INTO camera_congestion (camera_id, level, confidence, window_end) "
            "VALUES (:c, 'heavy', 0.9, :t)",
            {"c": CAM, "t": NOW},
        ),
        (
            "INSERT INTO anomalies (id, camera_id, scope, anomaly_type, metric_name, "
            "observed_value, "
            "score, threshold, window_start, window_end, detected_at, model_name, model_version, "
            "status, hints, severity) VALUES (:id, :c, 'camera', 'speed_drop', "
            "'loop_speed_kmh_rel', "
            "0.2, 4.0, 1.0, :ws, :we, :we, 'm', '1', 'active', '[\"speed_drop\"]', 'high')",
            {
                "id": ANOMALY,
                "c": CAM,
                "ws": NOW - timedelta(minutes=5),
                "we": NOW - timedelta(minutes=4),
            },
        ),
        (
            "INSERT INTO anomalies (id, camera_id, scope, affected_cameras, anomaly_type, "
            "metric_name, "
            "observed_value, score, threshold, window_start, window_end, detected_at, model_name, "
            "model_version, status, hints) VALUES (:id, NULL, 'system', CAST(:aff AS jsonb), "
            "'camera_degraded', 'frames', 0.5, 4, 3, :ws, :we, :we, 'system-health-rules', '1', "
            "'active', '[\"camera_degraded\"]')",
            {
                "id": SYSTEM,
                "aff": j([OTHER, "cam-elephant-castle", "cam-trafalgar-square"]),
                "ws": NOW - timedelta(minutes=2),
                "we": NOW - timedelta(minutes=1),
            },
        ),
        (
            "INSERT INTO incidents (id, camera_id, status, incident_type, version, severity, "
            "fused_confidence, confidence_components, start_time, suspected_at, confirmed_at, "
            "title, details) VALUES (:id, :c, 'confirmed', 'accident', 2, 'high', 0.9, "
            "CAST(:comp AS jsonb), :s, :s, :s, 'Accident at Old Street', CAST(:d AS jsonb))",
            {
                "id": INCIDENT,
                "c": CAM,
                "comp": j(components),
                "s": NOW - timedelta(minutes=5),
                "d": j(
                    {
                        "state": state,
                        "history": history,
                        "resolution": None,
                        "at_confirmation": {"rule": "both modalities agree"},
                    }
                ),
            },
        ),
        (
            "INSERT INTO incidents (id, camera_id, status, incident_type, version, severity, "
            "fused_confidence, start_time, suspected_at, end_time, title) VALUES (:id, :c, "
            "'resolved', 'road_works', 3, 'medium', 0.7, :s, :s, :e, 'Road works')",
            {
                "id": uuid.UUID(int=102),
                "c": OTHER,
                "s": NOW - timedelta(hours=2),
                "e": NOW - timedelta(hours=1),
            },
        ),
        (
            "INSERT INTO incident_evidence (incident_id, evidence_type, anomaly_id, weight) "
            "VALUES (:i, 'anomaly', :a, NULL)",
            {"i": INCIDENT, "a": ANOMALY},
        ),
        (
            "INSERT INTO summaries (id, scope, camera_id, incident_id, incident_version, "
            "period_start, "
            "period_end, headline, text, generation, model_name, model_version, latency_ms, "
            "faithfulness) VALUES (:id, 'incident', :c, :i, 2, :s, :e, "
            "'High accident at Old Street',"
            " 'Confirmed accident.', 'template', 'incident-summary-template', 'template-1', 0.1, "
            '\'{"fallback_reason": "template configuration"}\')',
            {
                "id": uuid.uuid4(),
                "c": CAM,
                "i": INCIDENT,
                "s": NOW - timedelta(minutes=5),
                "e": NOW,
            },
        ),
        (
            "INSERT INTO model_versions (model_name, version, stage, mlflow_run_id, metrics) "
            "VALUES "
            "('incident-text-classifier', '1', 'production', 'run-1', "
            '\'{"role": "classifier"}\'), '
            "('incident-summarizer', '3', 'production', NULL, '{}'), "
            "('incident-summarizer', '2', 'archived', NULL, '{}')",
            {},
        ),
        (
            "INSERT INTO scenario_runs (id, scenario_name, camera_id, incident_type, "
            "peak_severity, "
            "status, triggered_by, time_scale, started_at, expected_end_at) VALUES (:id, "
            "'flooding', :c, 'weather_hazard', 'critical', 'completed', 'cli', 6, :s, :e)",
            {
                "id": uuid.uuid4(),
                "c": CAM,
                "s": NOW - timedelta(hours=3),
                "e": NOW - timedelta(hours=3) + timedelta(minutes=4),
            },
        ),
    ]
    with engine.begin() as conn:
        for sql, params in stmts:
            conn.execute(text(sql), params)
        for i, r in enumerate([*REPORTS, UNLOCATED]):
            conn.execute(
                text(
                    "INSERT INTO reports (id, source, text, reported_at) VALUES (:id, :s, :t, :at)"
                ),
                {
                    "id": r,
                    "s": ["citizen", "police", "social_media"][i],
                    "t": f"report {i}",
                    "at": NOW - timedelta(minutes=4 - i),
                },
            )
            conn.execute(
                text(
                    "INSERT INTO nlp_results (report_id, camera_id, incident_type, "
                    "incident_confidence, severity, model_name, model_version, processed_at, "
                    "entities, stage) VALUES (:r, :c, :t, 0.95, 'high', 'm', '1', :at, "
                    "CAST(:e AS jsonb), 'active')"
                ),
                {
                    "r": r,
                    "c": None if r == UNLOCATED else CAM,
                    "t": "debris" if r == UNLOCATED else "accident",
                    "at": NOW - timedelta(minutes=3),
                    "e": j([{"text": "Brook Drive", "label": "LOCATION", "start": 0, "end": 11}]),
                },
            )
        for k in range(24):  # two minutes of 5 s windows on CAM
            end = NOW - timedelta(seconds=5 * k)
            conn.execute(
                text(
                    "INSERT INTO traffic_metrics (message_id, camera_id, window_start, window_end, "
                    "vehicle_count, avg_speed_kmh, stopped_vehicles, frames) VALUES "
                    "(:m, :c, :s, :e, :v, :sp, :st, 24)"
                ),
                {
                    "m": METRIC_MESSAGE if k == 3 else uuid.uuid4(),
                    "c": CAM,
                    "s": end - timedelta(seconds=5),
                    "e": end,
                    "v": 4 + k % 3,
                    "sp": 20.0 + k,
                    "st": k % 2,
                },
            )
            conn.execute(
                text(
                    "INSERT INTO congestion_history (camera_id, window_end, level) "
                    "VALUES (:c, :e, :l)"
                ),
                {"c": CAM, "e": end, "l": "heavy" if k < 18 else "moderate"},
            )


@pytest.fixture(scope="module")
def client(scratch_database_module: str, kafka_bootstrap: str, tmp_path_factory):
    from alembic import command

    from traffic_common.db import create_db_engine
    from traffic_common.db.migrate import alembic_config
    from traffic_common.db.seed import seed_cameras

    command.upgrade(alembic_config(scratch_database_module), "head")
    engine = create_db_engine(scratch_database_module)
    seed_cameras(engine)
    seed(engine)
    engine.dispose()
    frames = tmp_path_factory.mktemp("annotated")
    (frames / CAM).mkdir()
    (frames / CAM / "latest.jpg").write_bytes(JPEG)
    config = ApiConfig(
        annotated_dir=frames,
        scenarios_dir=ROOT / "services" / "event_simulator" / "scenarios",
        mjpeg_max_fps=50,
    )
    producer = FakeProducer()
    app = create_app(
        config,
        scratch_database_module,
        kafka_bootstrap,
        "http://mlflow.test",
        producer=producer,
        live_updates=False,
        mlflow_transport=httpx.MockTransport(mlflow_handler),
    )
    with TestClient(app) as c:
        c.producer = producer
        yield c


# ----------------------------------------------------------------------------- health


def test_health_and_ready(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200, r.json()
    assert r.json()["checks"] == {"postgres": "ok", "kafka": "ok", "live_consumer": "disabled"}


def test_cors_allows_the_dashboard(client) -> None:
    r = client.options(
        "/api/v1/cameras",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    r = client.get("/api/v1/cameras", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


# ----------------------------------------------------------------------------- cameras


def test_cameras_list(client) -> None:
    page = client.get("/api/v1/cameras").json()
    assert page["total"] == 4 and page["page"] == 1
    cams = {c["id"]: c for c in page["items"]}
    old = cams[CAM]
    assert old["congestion"]["level"] == "heavy" and old["open_incidents"] == 1
    assert old["latest_metrics"]["window_end"].startswith(NOW.isoformat()[:19])
    assert old["latest_metrics"]["vehicle_count"] == 4 and not old["vision_degraded"]
    assert old["frame_url"] == f"/api/v1/frames/{CAM}/latest.jpg"
    hyde = cams[OTHER]
    assert hyde["vision_degraded"] and hyde["vision_degraded_reasons"] == [
        "system-wide degradation"
    ]
    assert hyde["latest_metrics"] is None and hyde["open_incidents"] == 0


def test_camera_detail_aggregates_metrics_in_sql(client) -> None:
    d = client.get(f"/api/v1/cameras/{CAM}", params={"minutes": 10, "resolution": 60}).json()
    assert d["resolution_seconds"] == 60 and d["zone_config"] is not None
    assert sum(b["windows"] for b in d["metrics"]) == 24  # every window in some bucket
    assert all(b["avg_speed_kmh"] is not None for b in d["metrics"])
    assert max(b["stopped_vehicles_max"] for b in d["metrics"]) == 1
    assert [a["anomaly"]["anomaly_id"] for a in d["anomalies"]] == [str(ANOMALY)]
    hyde = client.get(f"/api/v1/cameras/{OTHER}").json()
    assert [a["anomaly"]["scope"] for a in hyde["anomalies"]] == ["system"]
    assert client.get("/api/v1/cameras/cam-nowhere").status_code == 404


# ----------------------------------------------------------------------------- incidents


def test_incident_list_filters_and_pages(client) -> None:
    page = client.get("/api/v1/incidents").json()
    assert [i["incident_id"] for i in page["items"]] == [str(INCIDENT), str(uuid.UUID(int=102))]
    assert page["items"][0]["change_reasons"] == ["confirmed (both modalities agree)"]
    one = client.get("/api/v1/incidents", params={"status": "confirmed", "camera_id": CAM}).json()
    assert one["total"] == 1 and one["items"][0]["incident_type"] == "accident"
    assert client.get("/api/v1/incidents", params={"severity": "critical"}).json()["total"] == 0
    assert (
        client.get("/api/v1/incidents", params={"incident_type": "road_works"}).json()["total"] == 1
    )
    since = (NOW - timedelta(minutes=30)).isoformat()
    assert client.get("/api/v1/incidents", params={"since": since}).json()["total"] == 1
    capped = client.get("/api/v1/incidents", params={"page_size": 10_000}).json()
    assert capped["page_size"] == 200
    second = client.get("/api/v1/incidents", params={"page_size": 1, "page": 2}).json()
    assert second["total"] == 2 and len(second["items"]) == 1
    assert client.get("/api/v1/incidents", params={"status": "nonsense"}).status_code == 422


def test_incident_detail(client) -> None:
    d = client.get(f"/api/v1/incidents/{INCIDENT}").json()
    assert d["incident"]["confidence_components"]["fusion"]["fused"] == 0.9
    assert [v["version"] for v in d["versions"]] == [1, 2]
    assert d["at_confirmation"]["rule"] == "both modalities agree"
    ev = d["evidence"]
    assert [a["anomaly"]["anomaly_id"] for a in ev["anomalies"]] == [str(ANOMALY)]
    assert [r["report"]["report_id"] for r in ev["reports"]] == [str(r) for r in REPORTS]
    assert ev["reports"][0]["nlp"]["incident_type"] == "accident"
    assert [m["message_id"] for m in ev["metrics"]] == [str(METRIC_MESSAGE)]
    assert ev["weights"] == {f"anomaly:{ANOMALY}": None}
    [summary] = d["summaries"]
    assert summary["generation"] == "template" and summary["incident_version"] == 2
    assert client.get(f"/api/v1/incidents/{uuid.uuid4()}").status_code == 404


# ----------------------------------------------------------------------------- reports


def test_reports_with_nlp_and_unlocated(client) -> None:
    page = client.get("/api/v1/reports").json()
    assert page["total"] == 3
    assert page["items"][0]["report"]["report_id"] == str(UNLOCATED)  # newest first
    [unlocated] = client.get("/api/v1/reports", params={"unlocated": True}).json()["items"]
    assert unlocated["unlocated"] and unlocated["nlp"]["camera_id"] is None
    assert unlocated["nlp"]["entities"][0]["text"] == "Brook Drive"
    assert client.get("/api/v1/reports", params={"camera_id": CAM}).json()["total"] == 2
    assert client.get("/api/v1/reports", params={"incident_type": "debris"}).json()["total"] == 1


# ----------------------------------------------------------------------------- analytics


def test_analytics(client) -> None:
    volume = client.get("/api/v1/analytics/volume", params={"hours": 2}).json()
    assert {v["camera_id"] for v in volume} == {CAM}
    assert sum(v["windows"] for v in volume) == 24
    assert sum(v["vehicles"] for v in volume) == sum(4 + k % 3 for k in range(24))
    shares = client.get(
        "/api/v1/analytics/congestion", params={"hours": 1, "bucket_minutes": 60}
    ).json()
    counts = {k: sum(s["counts"][k] for s in shares) for k in ("heavy", "moderate", "free_flow")}
    assert counts == {"heavy": 18, "moderate": 6, "free_flow": 0}
    assert all(abs(sum(s["shares"].values()) - 1) < 1e-3 for s in shares)
    by = client.get("/api/v1/analytics/incidents", params={"hours": 24}).json()
    assert {(c["incident_type"], c["severity"], c["status"], c["count"]) for c in by} == {
        ("accident", "high", "confirmed", 1),
        ("road_works", "medium", "resolved", 1),
    }


# ----------------------------------------------------------------------------- models


def test_models_with_mlflow_metrics(client) -> None:
    models = {m["model_name"]: m for m in client.get("/api/v1/models").json()}
    assert set(models) == {"incident-text-classifier", "incident-summarizer"}
    assert models["incident-summarizer"]["version"] == "3"  # production, not the archived one
    text_model = models["incident-text-classifier"]["mlflow"]
    assert text_model["champion_version"] == "1"
    assert text_model["run"]["metrics"] == {"test_incident_type_macro_f1": 0.956}
    assert text_model["run"]["tags"] == {}  # mlflow.* system tags left out
    assert text_model["latest_run"]["metrics"]["online_all_incident_accuracy"] == 0.98
    assert models["incident-summarizer"]["mlflow"] == {"registered": False}


# ----------------------------------------------------------------------------- scenarios


def test_scenarios_list_runs_and_trigger(client) -> None:
    scenarios = {s["name"]: s for s in client.get("/api/v1/scenarios").json()}
    assert set(scenarios) == {
        "broken_down_lorry",
        "flooding",
        "multi_vehicle_accident",
        "road_works",
    }
    accident = scenarios["multi_vehicle_accident"]
    assert accident["incident_type"] == "accident" and accident["duration_seconds"] == 18 * 60
    assert [p["name"] for p in accident["phases"]][0] == "first_reports"
    runs = client.get("/api/v1/scenarios/runs").json()
    assert runs["total"] == 1 and runs["items"][0]["scenario_name"] == "flooding"

    bad = client.post("/api/v1/scenarios/volcano/trigger", json={"camera_id": CAM})
    assert bad.status_code == 404 and "unknown scenario" in bad.json()["error"]["message"]
    bad = client.post("/api/v1/scenarios/flooding/trigger", json={"camera_id": "cam-nowhere"})
    assert bad.status_code == 404 and "unknown camera" in bad.json()["error"]["message"]
    assert (
        client.post(
            "/api/v1/scenarios/flooding/trigger", json={"camera_id": CAM, "time_scale": 0}
        ).status_code
        == 422
    )
    assert client.producer.sent == []
    r = client.post("/api/v1/scenarios/flooding/trigger", json={"camera_id": CAM, "time_scale": 6})
    assert r.status_code == 202
    [(topic, command)] = client.producer.sent
    assert topic == "simulator.control" and isinstance(command, SimulatorCommand)
    assert (command.action, command.scenario, command.camera_id, command.time_scale) == (
        "start_scenario",
        "flooding",
        CAM,
        6.0,
    )
    assert command.requested_by == "api" and r.json()["scenario_id"] == str(command.scenario_id)


# ----------------------------------------------------------------------------- frames


def test_latest_frame_is_not_cached(client) -> None:
    r = client.get(f"/api/v1/frames/{CAM}/latest.jpg")
    assert r.status_code == 200 and r.content == JPEG
    assert r.headers["content-type"] == "image/jpeg"
    assert "no-store" in r.headers["cache-control"] and "X-Frame-Time" in r.headers
    assert client.get(f"/api/v1/frames/{OTHER}/latest.jpg").status_code == 404
    assert client.get("/api/v1/frames/..%2F..%2Fetc/latest.jpg").status_code == 404


def test_mjpeg_stream(client) -> None:
    r = client.get(f"/api/v1/frames/{CAM}/stream.mjpg", params={"limit": 1, "fps": 50})
    assert r.status_code == 200
    assert r.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
    assert r.content.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    assert JPEG in r.content and f"Content-Length: {len(JPEG)}".encode() in r.content
    assert client.get(f"/api/v1/frames/{OTHER}/stream.mjpg").status_code == 404


# ----------------------------------------------------------------------------- pagination


def test_pagination_on_every_list(client) -> None:
    first = client.get("/api/v1/cameras", params={"page_size": 3}).json()
    second = client.get("/api/v1/cameras", params={"page_size": 3, "page": 2}).json()
    assert (first["total"], len(first["items"]), len(second["items"])) == (4, 3, 1)
    assert not {c["id"] for c in first["items"]} & {c["id"] for c in second["items"]}
    pages = [
        client.get("/api/v1/reports", params={"page_size": 1, "page": n}).json()["items"]
        for n in (1, 2, 3, 4)
    ]
    assert [len(p) for p in pages] == [1, 1, 1, 0]
    assert len({p[0]["report"]["report_id"] for p in pages[:3]}) == 3
    runs = client.get("/api/v1/scenarios/runs", params={"camera_id": OTHER}).json()
    assert runs["total"] == 0 and runs["items"] == []
    assert client.get("/api/v1/cameras", params={"page": 0}).status_code == 422


# ----------------------------------------------------------------------------- errors, logging


def test_errors_have_one_shape(client) -> None:
    missing = client.get(f"/api/v1/incidents/{uuid.uuid4()}")
    assert missing.status_code == 404
    err = missing.json()["error"]
    assert err["status"] == 404 and err["code"] == "not_found"
    assert "not found" in err["message"]
    assert err["request_id"] == missing.headers["X-Request-ID"]
    invalid = client.get("/api/v1/incidents", params={"status": "nonsense", "page_size": 0})
    err = invalid.json()["error"]
    assert invalid.status_code == 422 and err["code"] == "validation_error"
    assert {tuple(d["location"]) for d in err["details"]} == {
        ("query", "status"),
        ("query", "page_size"),
    }
    unknown = client.get("/api/v1/nothing-here").json()["error"]
    assert unknown["status"] == 404 and unknown["code"] == "not_found"


def test_unhandled_errors_are_500_in_the_same_shape(client) -> None:
    async def boom():
        raise RuntimeError("database exploded")

    client.app.add_api_route("/api/v1/test-boom", boom)
    raw = TestClient(client.app, raise_server_exceptions=False)
    r = raw.get("/api/v1/test-boom", headers={"X-Request-ID": "trace-42"})
    assert r.status_code == 500
    assert r.json() == {
        "error": {
            "status": 500,
            "code": "internal_error",
            "message": "internal error",
            "request_id": "trace-42",
        }
    }
    assert "database exploded" not in r.text  # no internals leak to the client


def test_requests_are_logged_with_timings(client, caplog) -> None:
    import logging

    with caplog.at_level(logging.DEBUG, logger="api_service.access"):
        r = client.get(
            "/api/v1/cameras", params={"page_size": 2}, headers={"X-Request-ID": "abc-1"}
        )
        client.get("/health")
    assert r.headers["X-Request-ID"] == "abc-1"
    assert r.headers["Server-Timing"].startswith("app;dur=")
    lines = [
        json.loads(rec.getMessage()) for rec in caplog.records if rec.name == "api_service.access"
    ]
    cameras = next(x for x in lines if x["path"] == "/api/v1/cameras")
    assert cameras["request_id"] == "abc-1" and cameras["status"] == 200
    assert cameras["method"] == "GET" and cameras["query"] == "page_size=2"
    assert cameras["duration_ms"] >= 0
    health = [rec for rec in caplog.records if "/health" in rec.getMessage()]
    assert health and all(rec.levelno == logging.DEBUG for rec in health)  # probes stay quiet
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) == 16
    # A hostile request id is replaced, not echoed into logs and headers.
    hostile = client.get("/health", headers={"X-Request-ID": "bad id <script>"})
    assert hostile.headers["X-Request-ID"] != "bad id <script>"
    assert len(hostile.headers["X-Request-ID"]) == 16


# ----------------------------------------------------------------------------- OpenAPI


def test_openapi_docs_have_tags_examples_and_errors(client) -> None:
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    tags = {t["name"] for t in spec["tags"]}
    assert {
        "cameras",
        "incidents",
        "reports",
        "analytics",
        "models",
        "scenarios",
        "frames",
        "health",
    } <= tags
    ops = [
        op for name, path in spec["paths"].items() if "test-" not in name for op in path.values()
    ]
    assert all(op.get("tags") for op in ops)
    trigger = spec["paths"]["/api/v1/scenarios/{scenario}/trigger"]["post"]
    assert trigger["responses"]["404"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ErrorResponse"
    )
    schemas = spec["components"]["schemas"]
    assert schemas["TriggerRequest"]["examples"][0]["time_scale"] == 6
    assert schemas["ScenarioInfo"]["examples"][0]["name"] == "broken_down_lorry"
    assert "X-Request-ID" in spec["info"]["description"]
