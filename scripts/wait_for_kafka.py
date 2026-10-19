"""Wait until a Kafka broker can serve consumer groups and idempotent producers.

    python scripts/wait_for_kafka.py [--bootstrap-servers localhost:9092] [--timeout 120]

A broker that answers ApiVersions (what the container healthcheck checks) can still be
loading its group and transaction coordinators: a consumer group can't join and an idempotent
producer can't get its producer ID ("Coordinator load in progress"). Tests started then lose
their first messages. This probe passes only when a throwaway consumer group has joined,
been assigned its partition and committed an offset, and an idempotent producer has had a
message acknowledged. CI runs it in the test image before the integration tests.
"""

import argparse
import sys
import time
import uuid

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic

PROBE_TOPIC = "kafka.readiness"


def ensure_topic(bootstrap: str) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    future = admin.create_topics([NewTopic(PROBE_TOPIC, num_partitions=1, replication_factor=1)])
    try:
        future[PROBE_TOPIC].result(timeout=30)
    except KafkaException as exc:
        if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
            raise


def group_ready(bootstrap: str, deadline: float) -> None:
    assigned: list[TopicPartition] = []
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"readiness-{uuid.uuid4().hex}",
            "enable.auto.commit": False,
        }
    )
    try:
        consumer.subscribe([PROBE_TOPIC], on_assign=lambda c, parts: assigned.extend(parts))
        while not assigned:
            if time.monotonic() > deadline:
                raise TimeoutError("the probe consumer group was never assigned its partition")
            consumer.poll(0.5)
        while True:
            try:
                consumer.commit(offsets=[TopicPartition(PROBE_TOPIC, 0, 0)], asynchronous=False)
                return
            except KafkaException as exc:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"the probe group could not commit: {exc}") from exc
                time.sleep(0.5)
    finally:
        consumer.close()


def producer_ready(bootstrap: str, deadline: float) -> None:
    errors: list[KafkaError] = []
    producer = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True, "acks": "all"})

    def delivered(err: KafkaError | None, _msg: object) -> None:
        if err is not None:
            errors.append(err)

    producer.produce(PROBE_TOPIC, b"ready", on_delivery=delivered)
    if producer.flush(max(deadline - time.monotonic(), 1)) > 0:
        raise TimeoutError("the idempotent probe producer's message was never acknowledged")
    if errors:
        raise RuntimeError(f"the probe message failed: {errors[0]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds")
    args = parser.parse_args(argv)
    started = time.monotonic()
    deadline = started + args.timeout
    try:
        ensure_topic(args.bootstrap_servers)
        group_ready(args.bootstrap_servers, deadline)
        print(f"consumer group joined and committed after {time.monotonic() - started:.1f} s")
        producer_ready(args.bootstrap_servers, deadline)
    except (TimeoutError, RuntimeError, KafkaException) as exc:
        print(f"Kafka at {args.bootstrap_servers} not ready: {exc}", file=sys.stderr)
        return 1
    print(f"Kafka ready after {time.monotonic() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
