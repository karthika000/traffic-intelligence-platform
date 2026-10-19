"""Typed Kafka producer and consumer built on confluent-kafka.

TypedProducer
    Validates every message against the topic's schema before sending, keys it by
    ``camera_id`` (when the message has one) and flushes when the process shuts down.

TypedConsumer
    At-least-once processing for one consumer group: offsets are committed manually,
    only after the handler succeeds or the message has been written to the group's
    dead-letter topic (``dlq.<group>``). Handler failures are retried with exponential
    backoff; messages that fail validation go to the dead-letter topic immediately.
    Stops on SIGINT/SIGTERM via the shared shutdown event in ``traffic_common.lifecycle``.
    Every poll beats the heartbeat ``consumer-<group>`` (``traffic_common.health``) with the
    number of partitions assigned, so the container healthcheck sees a consumer that is
    polling and owns partitions, not just a live process.
    With ``batch_handler`` and ``max_batch`` > 1, messages that are already waiting are handed
    over together (up to max_batch, never waiting to fill a batch); if a batch fails, its
    messages are retried one by one with the usual retries and dead-lettering.

Both record Prometheus metrics (``traffic_common.telemetry``): messages produced and consumed,
handler time, message age on arrival, errors by stage, and consumer lag per partition (from
librdkafka statistics every ``stats_interval_ms``).
"""

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException, Message, Producer
from pydantic import BaseModel, ValidationError

from traffic_common.health import heartbeat
from traffic_common.lifecycle import (
    install_signal_handlers,
    on_shutdown,
    run_shutdown_callbacks,
    shutdown_event,
)
from traffic_common.schemas import TOPIC_SCHEMAS, KafkaMessage, utcnow
from traffic_common.telemetry import (
    CONSUMER_LAG,
    MESSAGE_AGE_SECONDS,
    MESSAGE_ERRORS,
    MESSAGES_CONSUMED,
    MESSAGES_PRODUCED,
    PROCESSING_SECONDS,
    record_kafka_stats,
)
from traffic_common.topics import dead_letter_topic

log = logging.getLogger(__name__)

SchemaMap = Mapping[str, type[KafkaMessage]]
Handler = Callable[[KafkaMessage], None]
BatchHandler = Callable[[list[KafkaMessage]], None]

_MAX_ERROR_HEADER = 2_000


def _topic(msg: Message) -> str:
    """A consumed or delivered message always has a topic; the type allows None."""
    return msg.topic() or ""


def _schema(schemas: SchemaMap, topic: str) -> type[KafkaMessage]:
    try:
        return schemas[topic]
    except KeyError:
        raise KeyError(f"no schema registered for topic {topic!r}") from None


class TypedProducer:
    def __init__(
        self,
        bootstrap_servers: str,
        *,
        client_id: str = "traffic-common",
        schemas: SchemaMap = TOPIC_SCHEMAS,
        flush_timeout: float = 10.0,
        register_shutdown: bool = True,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self._schemas = schemas
        self._flush_timeout = flush_timeout
        self._closed = False
        self.delivery_errors = 0
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "client.id": client_id,
                "enable.idempotence": True,
                "acks": "all",
                "linger.ms": 5,
                **(config or {}),
            }
        )
        if register_shutdown:
            on_shutdown(self.close)

    def validate(self, topic: str, message: KafkaMessage | Mapping[str, Any]) -> KafkaMessage:
        schema = _schema(self._schemas, topic)
        if isinstance(message, BaseModel):
            if not isinstance(message, schema):
                raise TypeError(
                    f"topic {topic!r} expects {schema.__name__}, got {type(message).__name__}"
                )
            # Re-validate: model_construct() or a subclass could have bypassed validation.
            return schema.model_validate(message.model_dump())
        return schema.model_validate(message)

    def send(
        self,
        topic: str,
        message: KafkaMessage | Mapping[str, Any],
        *,
        key: str | None = None,
    ) -> KafkaMessage:
        """Validate and enqueue a message. Returns the validated model.

        Raises ``ValidationError``/``TypeError``/``KeyError`` before anything is sent.
        Delivery is asynchronous; failures are logged and counted in ``delivery_errors``.
        """
        validated = self.validate(topic, message)
        if key is None:
            key = getattr(validated, "camera_id", None)
        headers: list[tuple[str, str | bytes | None]] = [
            ("schema", type(validated).__name__.encode()),
            ("schema_version", str(validated.schema_version).encode()),
        ]
        value = validated.model_dump_json().encode()
        encoded_key = key.encode() if key else None

        def produce() -> None:
            self._producer.produce(
                topic, value=value, key=encoded_key, headers=headers, on_delivery=self._on_delivery
            )

        try:
            produce()
        except BufferError:
            # Local queue full: serve delivery callbacks to drain it, then retry once.
            self._producer.poll(1.0)
            produce()
        self._producer.poll(0)
        MESSAGES_PRODUCED.labels(topic).inc()
        return validated

    def flush(self, timeout: float | None = None) -> int:
        """Wait for outstanding messages. Returns how many are still undelivered."""
        remaining = self._producer.flush(self._flush_timeout if timeout is None else timeout)
        if remaining:
            log.warning("%d message(s) still undelivered after flush", remaining)
        return remaining

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.flush()

    def __enter__(self) -> "TypedProducer":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _on_delivery(self, err: KafkaError | None, msg: Message) -> None:
        if err is not None:
            self.delivery_errors += 1
            MESSAGE_ERRORS.labels(_topic(msg), "delivery").inc()
            log.error("delivery to %s failed: %s", msg.topic(), err)


@dataclass
class ConsumerStats:
    processed: int = 0
    retries: int = 0
    dead_lettered: int = 0
    batches: int = 0  # batches handled together (batch_handler)


class DeadLetterError(RuntimeError):
    """A message could not be written to the dead-letter topic; its offset was not committed."""


class TypedConsumer:
    def __init__(
        self,
        topics: Sequence[str],
        group_id: str,
        handler: Handler | None,
        bootstrap_servers: str,
        *,
        batch_handler: BatchHandler | None = None,
        max_batch: int = 1,
        schemas: SchemaMap = TOPIC_SCHEMAS,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
        max_backoff: float = 30.0,
        poll_timeout: float = 1.0,
        dead_letter_timeout: float = 10.0,
        auto_offset_reset: str = "earliest",
        stop_event: threading.Event | None = None,
        config: Mapping[str, Any] | None = None,
        stats_interval_ms: int = 15_000,
    ) -> None:
        for topic in topics:
            _schema(schemas, topic)
        self.topics = list(topics)
        self.group_id = group_id
        self.dead_letter_topic = dead_letter_topic(group_id)
        self.stats = ConsumerStats()
        if handler is None and batch_handler is None:
            raise ValueError("a handler or a batch_handler is required")
        self._batch_handler = batch_handler
        self._max_batch = max(1, max_batch) if batch_handler else 1
        if handler is None:
            assert batch_handler is not None
            one_by_one = batch_handler

            def handler(value: KafkaMessage) -> None:
                one_by_one([value])

        self._handler: Handler = handler
        self._schemas = schemas
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._poll_timeout = poll_timeout
        self._dead_letter_timeout = dead_letter_timeout
        self.health = heartbeat(f"consumer-{group_id}")
        self._assigned: set[tuple[str, int]] = set()
        self._uses_process_shutdown = stop_event is None
        self._stop = stop_event or shutdown_event()
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "enable.auto.commit": False,
                "auto.offset.reset": auto_offset_reset,
                "statistics.interval.ms": stats_interval_ms,
                "stats_cb": lambda stats: record_kafka_stats(group_id, stats),
                **(config or {}),
            }
        )
        self._dlq = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "client.id": f"{group_id}-dlq",
                "enable.idempotence": True,
                "acks": "all",
            }
        )

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """Consume until the stop event is set. Blocks the calling thread."""
        install_signal_handlers(log)
        self._consumer.subscribe(
            self.topics,
            on_assign=self._on_assign,
            on_revoke=self._on_revoke,
            on_lost=self._on_revoke,
        )
        log.info("consuming %s as group %s", self.topics, self.group_id)
        try:
            while not self._stop.is_set():
                self.health.beat(assigned=len(self._assigned))
                msg = self._consumer.poll(self._poll_timeout)
                if msg is None or not self._ok(msg):
                    continue
                if self._max_batch == 1:
                    self._process(msg)
                    continue
                # Whatever else is already waiting (timeout 0: never wait to fill a batch).
                more = self._consumer.consume(num_messages=self._max_batch - 1, timeout=0)
                self._process_batch([msg, *(m for m in more if self._ok(m))])
        finally:
            self._close()

    @property
    def assigned_partitions(self) -> int:
        return len(self._assigned)

    def _on_assign(self, _consumer: Consumer, partitions: list) -> None:
        self._assigned |= {(p.topic, p.partition) for p in partitions}
        log.info("group %s assigned %d partition(s)", self.group_id, len(self._assigned))
        self.health.beat(assigned=len(self._assigned))

    def _on_revoke(self, _consumer: Consumer, partitions: list) -> None:
        self._assigned -= {(p.topic, p.partition) for p in partitions}
        self.health.beat(assigned=len(self._assigned))
        for p in partitions:  # another consumer reports their lag now
            try:
                CONSUMER_LAG.remove(p.topic, str(p.partition), self.group_id)
            except KeyError:
                pass

    @staticmethod
    def _ok(msg: Message) -> bool:
        error = msg.error()
        if error is not None:
            if error.fatal():
                raise KafkaException(error)
            log.warning("consumer error: %s", error)
            return False
        return True

    def _process_batch(self, msgs: list[Message]) -> None:
        if len(msgs) == 1:
            self._process(msgs[0])
            return
        try:
            values = [self._schemas[_topic(m)].model_validate_json(m.value() or b"") for m in msgs]
        except ValidationError:
            values = None  # let the one-by-one path dead-letter the invalid message(s)
        if values is not None:
            for m, v in zip(msgs, values, strict=True):
                self._arrived(_topic(m), v)
            start = time.perf_counter()
            try:
                assert self._batch_handler is not None
                self._batch_handler(values)
            except Exception as exc:
                for m in msgs:
                    MESSAGE_ERRORS.labels(_topic(m), "handler").inc()
                log.warning(
                    "batch of %d failed (%s); processing its messages one by one", len(msgs), exc
                )
            else:
                each = (time.perf_counter() - start) / len(msgs)
                for m in msgs:
                    PROCESSING_SECONDS.labels(_topic(m), self.group_id).observe(each)
                last: dict[tuple[str, int | None], Message] = {}
                for m in msgs:
                    last[(_topic(m), m.partition())] = m  # offsets only grow per partition
                for m in last.values():
                    self._commit(m)
                self.stats.processed += len(msgs)
                self.stats.batches += 1
                return
        for m in msgs:
            if self._stop.is_set():
                return  # uncommitted: redelivered after restart
            self._process(m, arrived=values is not None)

    def _arrived(self, topic: str, value: KafkaMessage) -> None:
        MESSAGES_CONSUMED.labels(topic, self.group_id).inc()
        age = (utcnow() - value.timestamp).total_seconds()
        MESSAGE_AGE_SECONDS.labels(topic, self.group_id).observe(max(age, 0.0))

    def _process(self, msg: Message, arrived: bool = False) -> None:
        topic = _topic(msg)
        schema = self._schemas[topic]
        try:
            value = schema.model_validate_json(msg.value() or b"")
        except ValidationError as exc:
            MESSAGE_ERRORS.labels(topic, "validation").inc()
            self._dead_letter(msg, "validation", exc, attempts=0)
            self._commit(msg)
            return
        if not arrived:  # a batch that fell back to one by one counted its messages already
            self._arrived(topic, value)

        attempt = 0
        while True:
            attempt += 1
            start = time.perf_counter()
            try:
                self._handler(value)
            except Exception as exc:
                MESSAGE_ERRORS.labels(topic, "handler").inc()
                if attempt > self._max_retries:
                    self._dead_letter(msg, "processing", exc, attempts=attempt)
                    self._commit(msg)
                    return
                delay = min(self._initial_backoff * 2 ** (attempt - 1), self._max_backoff)
                self.stats.retries += 1
                log.warning(
                    "handler failed on %s[%d]@%d (attempt %d/%d), retrying in %.2fs: %s",
                    msg.topic(),
                    msg.partition(),
                    msg.offset(),
                    attempt,
                    self._max_retries + 1,
                    delay,
                    exc,
                )
                if self._stop.wait(delay):
                    # Shutting down mid-retry: leave the offset uncommitted so it is redelivered.
                    log.info(
                        "shutdown during retry; %s@%d will be redelivered",
                        msg.topic(),
                        msg.offset(),
                    )
                    return
            else:
                PROCESSING_SECONDS.labels(topic, self.group_id).observe(time.perf_counter() - start)
                self._commit(msg)
                self.stats.processed += 1
                return

    def _dead_letter(self, msg: Message, reason: str, error: Exception, attempts: int) -> None:
        original = msg.headers() or []  # a list on consumed messages; the type also allows a dict
        items = original.items() if isinstance(original, dict) else original
        headers: list[tuple[str, str | bytes | None]] = [(k, v) for k, v in items]
        headers += [
            (k, str(v).encode())
            for k, v in {
                "dlq.reason": reason,
                "dlq.error": f"{type(error).__name__}: {error}"[:_MAX_ERROR_HEADER],
                "dlq.original_topic": msg.topic(),
                "dlq.original_partition": msg.partition(),
                "dlq.original_offset": msg.offset(),
                "dlq.consumer_group": self.group_id,
                "dlq.attempts": attempts,
                "dlq.failed_at": utcnow().isoformat(),
            }.items()
        ]
        failures: list[KafkaError] = []

        def _on_delivery(err: KafkaError | None, _msg: Message) -> None:
            if err is not None:
                failures.append(err)

        self._dlq.produce(
            self.dead_letter_topic,
            value=msg.value(),
            key=msg.key(),
            headers=headers,
            on_delivery=_on_delivery,
        )
        remaining = self._dlq.flush(self._dead_letter_timeout)
        if remaining or failures:
            raise DeadLetterError(
                f"could not write {msg.topic()}@{msg.offset()} to {self.dead_letter_topic}: "
                f"{failures[0] if failures else 'flush timed out'}"
            )
        self.stats.dead_lettered += 1
        MESSAGE_ERRORS.labels(_topic(msg), "dead_letter").inc()
        log.error(
            "sent %s[%d]@%d to %s (%s): %s",
            msg.topic(),
            msg.partition(),
            msg.offset(),
            self.dead_letter_topic,
            reason,
            error,
        )

    def _commit(self, msg: Message) -> None:
        try:
            self._consumer.commit(message=msg, asynchronous=False)
        except KafkaException as exc:
            # Typically a rebalance; the new owner will reprocess (at-least-once).
            log.warning("commit of %s@%d failed: %s", msg.topic(), msg.offset(), exc)

    def _close(self) -> None:
        try:
            self._dlq.flush(self._dead_letter_timeout)
            self._consumer.close()
        finally:
            if self._uses_process_shutdown:
                run_shutdown_callbacks(log)
        log.info("consumer %s stopped (%s)", self.group_id, self.stats)
