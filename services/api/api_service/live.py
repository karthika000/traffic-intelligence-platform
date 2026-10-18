"""Live updates over a WebSocket.

A background thread consumes incidents, summaries, anomalies, traffic.metrics, nlp.results and
drift.alerts with a consumer group unique to this API process, starting at the latest offsets
(clients get what happens from now on; history is in the REST API). Each message becomes one
JSON envelope

    {"type": "<topic>", "camera_ids": [...], "ts": "...", "data": {...shared schema...}}

serialised once and fanned out to the clients whose subscription matches. traffic.metrics is
throttled per camera. Every client has its own bounded queue and sender task, so a slow client
only loses its own oldest messages (it is told how many) and never holds up the others; one
that keeps falling behind is disconnected. Idle connections get heartbeats.

Client protocol (all optional): ``?topics=incidents,summaries&cameras=cam-a`` on connect, or
later ``{"action": "subscribe", "topics": [...], "cameras": [...]}``; ``{"action": "ping"}``.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

from confluent_kafka import Consumer, KafkaException
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from traffic_common.schemas import TOPIC_SCHEMAS
from traffic_common.topics import Topic

log = logging.getLogger(__name__)
LIVE_TOPICS = (
    Topic.INCIDENTS,
    Topic.SUMMARIES,
    Topic.ANOMALIES,
    Topic.TRAFFIC_METRICS,
    Topic.NLP_RESULTS,
    Topic.DRIFT_ALERTS,
)
router = APIRouter(tags=["live"])


def camera_ids(value) -> list[str]:
    cams = list(getattr(value, "affected_cameras", None) or [])
    cam = getattr(value, "camera_id", None)
    return [cam, *cams] if cam else cams


@dataclass(eq=False)  # identity: clients live in a set
class Client:
    queue: asyncio.Queue
    topics: set[str] | None = None  # None: everything
    cameras: set[str] | None = None
    dropped: int = 0  # since the last notice
    dropped_in_a_row: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def wants(self, topic: str, cams: list[str]) -> bool:
        if self.topics is not None and topic not in self.topics:
            return False
        # Messages without a camera (e.g. unlocated reports) go to camera-filtered clients too.
        return self.cameras is None or not cams or any(c in self.cameras for c in cams)


class LiveHub:
    def __init__(
        self,
        bootstrap: str,
        queue_size: int = 500,
        metrics_interval: float = 2.0,
        max_drops: int = 5000,
        topics: tuple[str, ...] = LIVE_TOPICS,
    ) -> None:
        self.bootstrap, self.topics = bootstrap, [str(t) for t in topics]
        self.queue_size, self.metrics_interval, self.max_drops = (
            queue_size,
            metrics_interval,
            max_drops,
        )
        self.group = f"api-live-{uuid.uuid4().hex[:12]}"
        self.clients: set[Client] = set()
        self.running = False
        self.assigned = threading.Event()  # partitions assigned: messages from now on arrive
        self.last_poll = 0.0  # monotonic
        self.error: str | None = None
        self.stats = {"received": 0, "sent": 0, "dropped": 0, "throttled": 0, "invalid": 0}
        self._last_metrics: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ consumer thread

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._thread = threading.Thread(target=self._run, name="live-consumer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def healthy(self) -> bool:
        return self.running and time.monotonic() - self.last_poll < 30

    def _run(self) -> None:
        consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap,
                "group.id": self.group,
                "auto.offset.reset": "latest",
                "enable.auto.commit": False,  # a throwaway group: nothing to resume
            }
        )
        try:
            consumer.subscribe(self.topics, on_assign=lambda c, parts: self.assigned.set())
            self.running = True
            log.info("live consumer %s on %s", self.group, self.topics)
            while not self._stop.is_set():
                msg = consumer.poll(0.5)
                self.last_poll = time.monotonic()
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().fatal():
                        raise KafkaException(msg.error())
                    continue
                self._handle(msg.topic(), msg.value())
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("live consumer stopped")
        finally:
            self.running = False
            consumer.close()

    def _handle(self, topic: str, raw: bytes | None) -> None:
        self.stats["received"] += 1
        try:
            value = TOPIC_SCHEMAS[topic].model_validate_json(raw or b"")
        except ValidationError:
            self.stats["invalid"] += 1
            return
        cams = camera_ids(value)
        if topic == Topic.TRAFFIC_METRICS and cams:
            now = time.monotonic()
            if now - self._last_metrics.get(cams[0], 0.0) < self.metrics_interval:
                self.stats["throttled"] += 1
                return
            self._last_metrics[cams[0]] = now
        text = json.dumps(
            {
                "type": topic,
                "camera_ids": cams,
                "ts": value.timestamp.isoformat(),
                "data": value.model_dump(mode="json"),
            }
        )
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.publish, topic, cams, text)

    # ------------------------------------------------------------------ fan-out (event loop)

    def register(self, topics: set[str] | None, cameras: set[str] | None) -> Client:
        client = Client(asyncio.Queue(self.queue_size), topics, cameras)
        self.clients.add(client)
        return client

    def unregister(self, client: Client) -> None:
        self.clients.discard(client)

    def publish(self, topic: str, cams: list[str], text: str) -> None:
        """Never blocks: a full client queue loses its oldest message instead."""
        for client in list(self.clients):
            if not client.wants(topic, cams):
                continue
            if client.queue.full():
                client.queue.get_nowait()
                client.dropped += 1
                client.dropped_in_a_row += 1
                self.stats["dropped"] += 1
            client.queue.put_nowait(text)


def _parse_set(value) -> set[str] | None:
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else list(value)
    cleaned = {str(v).strip() for v in items if str(v).strip()}
    return cleaned or None


@router.websocket("/ws/live")
async def live(websocket: WebSocket) -> None:
    hub: LiveHub | None = getattr(websocket.app.state, "hub", None)
    await websocket.accept()
    if hub is None:
        await websocket.close(code=1013, reason="live updates are disabled")
        return
    params = websocket.query_params
    topics = _parse_set(params.get("topics"))
    unknown = (topics or set()) - set(hub.topics)
    if unknown:
        await websocket.close(code=1008, reason=f"unknown topics {sorted(unknown)}")
        return
    client = hub.register(topics, _parse_set(params.get("cameras")))
    cfg = websocket.app.state.config
    await websocket.send_json(
        {
            "type": "hello",
            "client_id": client.id,
            "topics": hub.topics,
            "subscribed": {
                "topics": sorted(client.topics or hub.topics),
                "cameras": sorted(client.cameras) if client.cameras else "all",
            },
            "metrics_interval_seconds": hub.metrics_interval,
            "heartbeat_seconds": cfg.ws_heartbeat_seconds,
        }
    )

    async def sender() -> None:
        while True:
            try:
                text = await asyncio.wait_for(client.queue.get(), cfg.ws_heartbeat_seconds)
            except TimeoutError:
                await websocket.send_json({"type": "heartbeat", "ts": time.time()})
                continue
            if client.dropped_in_a_row > hub.max_drops:
                await websocket.close(code=1013, reason="client too slow")
                return
            if client.dropped:
                await websocket.send_json({"type": "dropped", "count": client.dropped})
                client.dropped = 0
            await websocket.send_text(text)
            client.dropped_in_a_row = 0
            hub.stats["sent"] += 1

    async def receiver() -> None:
        while True:
            message = await websocket.receive_json()
            action = message.get("action")
            if action == "subscribe":
                new_topics = _parse_set(message.get("topics"))
                bad = (new_topics or set()) - set(hub.topics)
                if bad:
                    await websocket.send_json(
                        {"type": "error", "error": f"unknown topics {sorted(bad)}"}
                    )
                    continue
                client.topics, client.cameras = new_topics, _parse_set(message.get("cameras"))
                await websocket.send_json(
                    {
                        "type": "subscribed",
                        "topics": sorted(client.topics or hub.topics),
                        "cameras": sorted(client.cameras) if client.cameras else "all",
                    }
                )
            elif action == "ping":
                await websocket.send_json({"type": "pong", "ts": time.time()})
            else:
                await websocket.send_json({"type": "error", "error": f"unknown action {action!r}"})

    tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            exc = t.exception()
            if exc and not isinstance(exc, WebSocketDisconnect | RuntimeError):
                log.warning("live client %s: %s", client.id, exc)
    finally:
        for t in tasks:
            t.cancel()
        hub.unregister(client)
