"""Producer/consumer tests against a real Kafka broker.

Each test gets its own throwaway topic and consumer group (plus that group's DLQ topic),
created and deleted here, so the platform's topics are never touched.
Skipped when Kafka isn't reachable or confluent-kafka can't load (see conftest).
"""

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from conftest import SMART_APP_CONTROL_HINT

ck = pytest.importorskip("confluent_kafka", reason=SMART_APP_CONTROL_HINT, exc_type=ImportError)
from confluent_kafka import Consumer, Producer, TopicPartition  # noqa: E402
from confluent_kafka.admin import AdminClient, NewTopic  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from traffic_common.lifecycle import on_shutdown, run_shutdown_callbacks  # noqa: E402
from traffic_common.messaging import TypedConsumer, TypedProducer  # noqa: E402
from traffic_common.schemas import KafkaMessage, TextReport, TrafficMetrics  # noqa: E402

T0 = datetime(2026, 10, 14, 8, 0, tzinfo=UTC)


def metrics(camera_id: str = "cam-a", count: int = 10) -> TrafficMetrics:
    return TrafficMetrics(
        camera_id=camera_id,
        window_start=T0,
        window_end=T0 + timedelta(minutes=1),
        vehicle_count=count,
    )


@dataclass
class Scratch:
    bootstrap: str
    topic: str
    group: str
    dlq: str

    @property
    def schemas(self) -> dict[str, type[KafkaMessage]]:
        return {self.topic: TrafficMetrics}

    def producer(self, **kwargs) -> TypedProducer:
        kwargs.setdefault("register_shutdown", False)
        return TypedProducer(self.bootstrap, schemas=self.schemas, **kwargs)

    def consumer(self, handler, stop: threading.Event, **kwargs) -> TypedConsumer:
        kwargs.setdefault("initial_backoff", 0.05)
        return TypedConsumer(
            [self.topic],
            self.group,
            handler,
            self.bootstrap,
            schemas=self.schemas,
            stop_event=stop,
            poll_timeout=0.2,
            **kwargs,
        )

    def raw_produce(self, *values: bytes, key: bytes | None = b"cam-a") -> None:
        p = Producer({"bootstrap.servers": self.bootstrap})
        for v in values:
            p.produce(self.topic, value=v, key=key)
        assert p.flush(10) == 0

    def read(self, topic: str, expected: int, timeout: float = 15.0) -> list:
        c = Consumer(
            {
                "bootstrap.servers": self.bootstrap,
                "group.id": f"reader-{uuid.uuid4().hex}",
                "enable.auto.commit": False,
            }
        )
        c.assign([TopicPartition(topic, 0, 0)])
        out, deadline = [], time.monotonic() + timeout
        while len(out) < expected and time.monotonic() < deadline:
            m = c.poll(0.5)
            if m is not None and not m.error():
                out.append(m)
        time.sleep(0.5)  # catch anything beyond `expected`
        while (m := c.poll(0.2)) is not None:
            if not m.error():
                out.append(m)
        c.close()
        return out

    def committed_offset(self) -> int:
        c = Consumer({"bootstrap.servers": self.bootstrap, "group.id": self.group})
        [tp] = c.committed([TopicPartition(self.topic, 0)], timeout=10)
        c.close()
        return tp.offset


@pytest.fixture
def scratch(kafka_bootstrap: str) -> Iterator[Scratch]:
    suffix = uuid.uuid4().hex[:10]
    s = Scratch(kafka_bootstrap, f"test.metrics.{suffix}", f"test-{suffix}", f"dlq.test-{suffix}")
    admin = AdminClient({"bootstrap.servers": kafka_bootstrap})
    for f in admin.create_topics([NewTopic(t, 1, 1) for t in (s.topic, s.dlq)]).values():
        f.result(timeout=15)
    yield s
    for f in admin.delete_topics([s.topic, s.dlq]).values():
        f.result(timeout=15)


def run_until(
    consumer: TypedConsumer, stop: threading.Event, done: Callable[[], bool], timeout: float = 30.0
) -> None:
    errors: list[BaseException] = []

    def target() -> None:
        try:
            consumer.run()
        except BaseException as exc:  # surface thread failures in the test
            errors.append(exc)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline and t.is_alive():
        time.sleep(0.1)
    stop.set()
    t.join(timeout=15)
    assert not t.is_alive(), "consumer did not stop"
    if errors:
        raise errors[0]
    assert done(), "condition not reached before timeout"


def header(msg, name: str) -> str:
    return dict(msg.headers())[name].decode()


# ----------------------------------------------------------------------------- producer


def test_producer_validates_keys_by_camera_and_tags_schema(scratch: Scratch) -> None:
    with scratch.producer() as p:
        p.send(scratch.topic, metrics("cam-a", 1))
        p.send(
            scratch.topic,
            {
                "camera_id": "cam-b",
                "window_start": T0,
                "window_end": T0 + timedelta(minutes=1),
                "vehicle_count": 2,
            },
        )
        p.send(scratch.topic, metrics("cam-a", 3), key="override")

    got = scratch.read(scratch.topic, 3)
    assert [m.key() for m in got] == [b"cam-a", b"cam-b", b"override"]
    assert [TrafficMetrics.model_validate_json(m.value()).vehicle_count for m in got] == [1, 2, 3]
    assert header(got[0], "schema") == "TrafficMetrics"
    assert header(got[0], "schema_version") == "1"
    assert p.delivery_errors == 0


def test_producer_rejects_before_sending(scratch: Scratch) -> None:
    with scratch.producer() as p:
        with pytest.raises(ValidationError):
            p.send(scratch.topic, {"camera_id": "cam-a", "vehicle_count": -5})
        with pytest.raises(TypeError, match="expects TrafficMetrics"):
            p.send(scratch.topic, TextReport(source="citizen", text="hello"))
        with pytest.raises(KeyError, match="no schema"):
            p.send("not.registered", metrics())
        with pytest.raises(ValidationError):  # model_construct() skips validation; send() doesn't
            p.send(scratch.topic, TrafficMetrics.model_construct(camera_id="", vehicle_count=-1))
    assert scratch.read(scratch.topic, 0, timeout=2) == []


def test_producer_flushes_via_shutdown_callbacks(scratch: Scratch) -> None:
    p = scratch.producer(register_shutdown=True, config={"linger.ms": 60_000})
    p.send(scratch.topic, metrics())  # sits in the local queue because of the long linger
    run_shutdown_callbacks()
    assert len(scratch.read(scratch.topic, 1)) == 1


# ----------------------------------------------------------------------------- consumer


def test_consumer_processes_and_commits(scratch: Scratch) -> None:
    with scratch.producer() as p:
        for i in range(3):
            p.send(scratch.topic, metrics(count=i))
    seen: list[int] = []
    stop = threading.Event()
    c = scratch.consumer(lambda m: seen.append(m.vehicle_count), stop)
    run_until(c, stop, lambda: len(seen) == 3)

    assert seen == [0, 1, 2]
    assert c.stats.processed == 3 and c.stats.dead_lettered == 0
    assert scratch.committed_offset() == 3


def test_invalid_messages_go_to_dlq_and_are_committed(scratch: Scratch) -> None:
    good = metrics(count=99).model_dump_json().encode()
    wrong_shape = json.dumps({"camera_id": "cam-a", "vehicle_count": "lots"}).encode()
    scratch.raw_produce(b"not json", wrong_shape, good)
    seen: list[int] = []
    stop = threading.Event()
    c = scratch.consumer(lambda m: seen.append(m.vehicle_count), stop)
    run_until(c, stop, lambda: len(seen) == 1 and c.stats.dead_lettered == 2)

    assert seen == [99]
    assert scratch.committed_offset() == 3
    dlq = scratch.read(scratch.dlq, 2)
    assert [m.value() for m in dlq] == [b"not json", wrong_shape]
    assert dlq[0].key() == b"cam-a"
    assert {header(m, "dlq.reason") for m in dlq} == {"validation"}
    assert header(dlq[1], "dlq.original_topic") == scratch.topic
    assert header(dlq[1], "dlq.original_offset") == "1"
    assert header(dlq[1], "dlq.consumer_group") == scratch.group
    assert "ValidationError" in header(dlq[1], "dlq.error")


def test_processing_failures_retry_then_dead_letter(scratch: Scratch) -> None:
    with scratch.producer() as p:
        p.send(scratch.topic, metrics("cam-bad"))
        p.send(scratch.topic, metrics("cam-good"))
    calls: dict[str, int] = {}

    def handler(m: TrafficMetrics) -> None:
        calls[m.camera_id] = calls.get(m.camera_id, 0) + 1
        if m.camera_id == "cam-bad":
            raise RuntimeError("database unavailable")

    stop = threading.Event()
    c = scratch.consumer(handler, stop, max_retries=2)
    run_until(c, stop, lambda: c.stats.processed == 1 and c.stats.dead_lettered == 1)

    assert calls == {"cam-bad": 3, "cam-good": 1}  # 1 try + 2 retries
    assert c.stats.retries == 2
    assert scratch.committed_offset() == 2
    [dead] = scratch.read(scratch.dlq, 1)
    assert dead.key() == b"cam-bad"
    assert header(dead, "dlq.reason") == "processing"
    assert header(dead, "dlq.attempts") == "3"
    assert header(dead, "dlq.error") == "RuntimeError: database unavailable"
    assert header(dead, "schema") == "TrafficMetrics"  # original headers preserved


def test_transient_failure_recovers_without_dlq(scratch: Scratch) -> None:
    with scratch.producer() as p:
        p.send(scratch.topic, metrics())
    attempts = []

    def flaky(_m: TrafficMetrics) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("blip")

    stop = threading.Event()
    c = scratch.consumer(flaky, stop)
    run_until(c, stop, lambda: c.stats.processed == 1)

    assert len(attempts) == 2 and c.stats.dead_lettered == 0
    assert scratch.read(scratch.dlq, 0, timeout=2) == []


def test_shutdown_during_backoff_leaves_message_uncommitted(scratch: Scratch) -> None:
    with scratch.producer() as p:
        p.send(scratch.topic, metrics())
    stop = threading.Event()

    def always_fails(_m: TrafficMetrics) -> None:
        raise RuntimeError("down")

    c = scratch.consumer(always_fails, stop, initial_backoff=30.0)
    started = time.monotonic()
    run_until(c, stop, lambda: c.stats.retries == 1)

    assert time.monotonic() - started < 20, "stop should interrupt the backoff wait"
    assert scratch.committed_offset() < 0  # nothing committed: redelivered on restart
    assert scratch.read(scratch.dlq, 0, timeout=2) == []

    # A fresh consumer in the same group gets the message again.
    seen: list[int] = []
    stop2 = threading.Event()
    c2 = scratch.consumer(lambda m: seen.append(m.vehicle_count), stop2)
    run_until(c2, stop2, lambda: len(seen) == 1)


def test_consumer_rejects_topics_without_schema(kafka_bootstrap: str) -> None:
    with pytest.raises(KeyError, match="no schema"):
        TypedConsumer(["not.registered"], "g", lambda m: None, kafka_bootstrap)


def test_process_shutdown_runs_registered_callbacks(scratch: Scratch) -> None:
    """With the default (process-wide) stop event, run() ends by running shutdown callbacks."""
    ran = []
    on_shutdown(lambda: ran.append("flushed"))
    c = TypedConsumer(
        [scratch.topic],
        scratch.group,
        lambda m: None,
        scratch.bootstrap,
        schemas=scratch.schemas,
        poll_timeout=0.2,
    )
    threading.Timer(1.0, c.stop).start()
    c.run()
    assert ran == ["flushed"]
