"""Live WebSocket: fan-out logic (filters, throttling, slow clients) without Kafka, and an
integration test with a real consumer: publish to Kafka, receive the envelope over /ws/live."""

import asyncio
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from conftest import SMART_APP_CONTROL_HINT

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "services" / "api")]

pytest.importorskip("fastapi")
pytest.importorskip("confluent_kafka", reason=SMART_APP_CONTROL_HINT, exc_type=ImportError)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from api_service.live import Client, LiveHub, camera_ids  # noqa: E402
from traffic_common.schemas import Anomaly, Incident, TrafficMetrics  # noqa: E402
from traffic_common.topics import Topic  # noqa: E402

NOW = datetime.now(UTC)


def incident(camera: str) -> Incident:
    return Incident(
        camera_id=camera,
        severity="high",
        fused_confidence=0.8,
        start_time=NOW,
        title="Accident",
        incident_type="accident",
    )


def metrics(camera: str) -> TrafficMetrics:
    return TrafficMetrics(
        camera_id=camera, window_start=NOW - timedelta(seconds=5), window_end=NOW, vehicle_count=3
    )


# ----------------------------------------------------------------------------- fan-out


def test_subscription_filters() -> None:
    c = Client(asyncio.Queue(), topics={"incidents"}, cameras={"cam-a"})
    assert c.wants("incidents", ["cam-a"]) and not c.wants("incidents", ["cam-b"])
    assert not c.wants("summaries", ["cam-a"])
    assert c.wants("incidents", [])  # no camera (e.g. an unlocated report): still delivered
    assert Client(asyncio.Queue()).wants("nlp.results", ["anything"])
    system = Anomaly(
        scope="system",
        affected_cameras=["cam-a", "cam-b"],
        anomaly_type="camera_degraded",
        metric_name="frames",
        observed_value=0.5,
        score=4,
        threshold=3,
        model_name="m",
        model_version="1",
        window_start=NOW - timedelta(seconds=15),
        window_end=NOW,
    )
    assert camera_ids(system) == ["cam-a", "cam-b"]
    assert camera_ids(incident("cam-c")) == ["cam-c"]


def test_slow_client_loses_oldest_messages_only() -> None:
    hub = LiveHub("unused:9092", queue_size=2)
    slow = hub.register(None, None)
    fast = hub.register(None, None)
    for i in range(5):
        hub.publish("incidents", ["cam-a"], f"m{i}")
        if not fast.queue.empty():
            fast.queue.get_nowait()  # this client keeps up
    assert [slow.queue.get_nowait() for _ in range(2)] == ["m3", "m4"]  # newest kept
    assert slow.dropped == 3 and fast.dropped == 0 and hub.stats["dropped"] == 3


def test_metrics_are_throttled_per_camera() -> None:
    async def run() -> list[str]:
        hub = LiveHub("unused:9092", metrics_interval=60)
        hub._loop = asyncio.get_running_loop()
        client = hub.register(None, None)
        for camera in ("cam-a", "cam-a", "cam-b", "cam-a"):
            hub._handle(Topic.TRAFFIC_METRICS, metrics(camera).model_dump_json().encode())
        hub._handle(Topic.INCIDENTS, incident("cam-a").model_dump_json().encode())
        hub._handle(Topic.INCIDENTS, b"not json")  # skipped, counted
        await asyncio.sleep(0.05)
        assert hub.stats["throttled"] == 2 and hub.stats["invalid"] == 1
        return [client.queue.get_nowait() for _ in range(client.queue.qsize())]

    import json

    envelopes = [json.loads(t) for t in asyncio.run(run())]
    assert [(e["type"], e["camera_ids"]) for e in envelopes] == [
        ("traffic.metrics", ["cam-a"]),
        ("traffic.metrics", ["cam-b"]),
        ("incidents", ["cam-a"]),
    ]
    assert envelopes[2]["data"]["incident_type"] == "accident" and envelopes[2]["ts"]


# ----------------------------------------------------------------------------- integration


def test_live_websocket_end_to_end(kafka_bootstrap: str, scratch_database: str) -> None:
    from fastapi.testclient import TestClient

    from api_service.app import create_app
    from api_service.config import ApiConfig
    from traffic_common.messaging import TypedProducer

    config = ApiConfig(ws_heartbeat_seconds=1.0, ws_metrics_interval_seconds=30)
    app = create_app(config, scratch_database, kafka_bootstrap, "http://mlflow.test")
    mine, other = f"test-cam-{uuid.uuid4().hex[:6]}", f"test-cam-{uuid.uuid4().hex[:6]}"
    with TestClient(app) as client:
        hub = app.state.hub
        # Assigned means positioned at the end offsets: everything sent from now on arrives.
        assert hub.assigned.wait(60), "live consumer never got its partitions"
        ready = client.get("/ready").json()
        assert ready["checks"]["live_consumer"] == "ok", ready
        with client.websocket_connect(
            f"/ws/live?topics=incidents,traffic.metrics&cameras={mine}"
        ) as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert hello["subscribed"] == {
                "topics": ["incidents", "traffic.metrics"],
                "cameras": [mine],
            }
            with TypedProducer(kafka_bootstrap, register_shutdown=False) as p:
                p.send(Topic.INCIDENTS, incident(other))  # filtered out
                p.send(Topic.INCIDENTS, incident(mine))
                for _ in range(3):
                    p.send(Topic.TRAFFIC_METRICS, metrics(mine))  # throttled to one
            received, heartbeats = [], 0
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                msg = ws.receive_json()  # heartbeats arrive at least every second
                if msg["type"] == "heartbeat":
                    heartbeats += 1
                    if any(m["type"] == "incidents" for m in received) and heartbeats >= 2:
                        break
                    continue
                received.append(msg)
            ws.send_json({"action": "subscribe", "topics": ["summaries"]})
            reply = ws.receive_json()
            while reply["type"] == "heartbeat":
                reply = ws.receive_json()
            assert reply == {"type": "subscribed", "topics": ["summaries"], "cameras": "all"}
            ws.send_json({"action": "subscribe", "topics": ["nonsense"]})
            reply = ws.receive_json()
            while reply["type"] == "heartbeat":
                reply = ws.receive_json()
            assert reply["type"] == "error"
    types = [(m["type"], m["camera_ids"]) for m in received]
    assert ("incidents", [mine]) in types
    assert all(cams == [mine] for _, cams in types)  # nothing for the other camera
    assert types.count(("traffic.metrics", [mine])) == 1
    assert heartbeats >= 2
    [inc] = [m for m in received if m["type"] == "incidents"]
    assert inc["data"]["camera_id"] == mine and inc["data"]["incident_type"] == "accident"
