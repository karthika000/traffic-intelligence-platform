"""Kafka helpers for the integration tests: where a topic ends now, starting a new consumer
group there, and waiting until a consumer actually owns its partitions.

A new group with auto.offset.reset="latest" plus a fixed sleep is racy: on a freshly started
broker the group coordinator can take many seconds to load, and anything produced before the
group is assigned (and has resolved "latest") is silently skipped. Committing the group's
starting offsets first makes the start position independent of when the join completes, and
waiting for the assignment turns a slow coordinator into a clear failure instead of a lost
message.
"""

import time
import uuid
from collections.abc import Iterable

from confluent_kafka import Consumer, TopicPartition

from traffic_common.messaging import TypedConsumer

ASSIGNMENT_TIMEOUT = 60.0


def end_offsets(bootstrap: str, topic: str) -> list[TopicPartition]:
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": f"probe-{uuid.uuid4().hex}"})
    try:
        parts = c.list_topics(topic, timeout=10).topics[topic].partitions
        return [
            TopicPartition(
                topic, p, c.get_watermark_offsets(TopicPartition(topic, p), timeout=10)[1]
            )
            for p in parts
        ]
    finally:
        c.close()


def start_group_at_end(bootstrap: str, group: str, topics: Iterable[str]) -> None:
    """Commit the current end of `topics` for `group`, so its consumer starts exactly there
    (no replay of history, nothing skipped) however late its partitions are assigned."""
    seed = Consumer({"bootstrap.servers": bootstrap, "group.id": group})
    try:
        offsets = [tp for topic in topics for tp in end_offsets(bootstrap, topic)]
        seed.commit(offsets=offsets, asynchronous=False)
    finally:
        seed.close()


def wait_for_assignment(
    consumer: TypedConsumer, bootstrap: str, timeout: float = ASSIGNMENT_TIMEOUT
) -> None:
    """Block until `consumer` (running in another thread, alone in its group) owns every
    partition of its topics."""
    probe = Consumer({"bootstrap.servers": bootstrap, "group.id": f"probe-{uuid.uuid4().hex}"})
    try:
        meta = probe.list_topics(timeout=10)
        expected = sum(len(meta.topics[t].partitions) for t in consumer.topics)
    finally:
        probe.close()
    deadline = time.monotonic() + timeout
    while consumer.assigned_partitions < expected:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"group {consumer.group_id} has {consumer.assigned_partitions} of {expected} "
                f"partitions after {timeout:.0f} s (is Kafka's group coordinator ready?)"
            )
        time.sleep(0.1)
