"""Integration test: one CameraWorker on a short synthetic video publishes FrameMessages
to the real `video.frames` topic.

Messages are published under a unique `test-cam-...` camera id and read back from the
partition offsets recorded before the run, so they don't mix with live camera traffic.
They stay in `video.frames` until Kafka's retention removes them; their frame files are
written to a temp directory and are gone after the test, which downstream consumers
must tolerate anyway (see FrameMessage).
"""

import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from conftest import SMART_APP_CONTROL_HINT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "video_ingest"))

pytest.importorskip("confluent_kafka", reason=SMART_APP_CONTROL_HINT, exc_type=ImportError)
pytest.importorskip("cv2", exc_type=ImportError)
from confluent_kafka import Consumer, TopicPartition  # noqa: E402

from traffic_common.messaging import TypedProducer  # noqa: E402
from traffic_common.schemas import FrameMessage  # noqa: E402
from traffic_common.topics import Topic  # noqa: E402
from video_ingest.config import IngestConfig  # noqa: E402
from video_ingest.sources import SourceKind, VideoSource  # noqa: E402
from video_ingest.storage import FrameStore  # noqa: E402
from video_ingest.synthetic import generate_traffic_video  # noqa: E402
from video_ingest.worker import CameraWorker  # noqa: E402

SOURCE_FPS = 10.0
VIDEO_SECONDS = 2.0  # short, so the 6 s run loops the file twice
SAMPLE_FPS = 4.0
RUN_SECONDS = 6.0


def end_offsets(bootstrap: str, topic: str) -> list[TopicPartition]:
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": f"probe-{uuid.uuid4().hex}"})
    try:
        partitions = c.list_topics(topic, timeout=10).topics[topic].partitions
        out = []
        for p in partitions:
            _low, high = c.get_watermark_offsets(TopicPartition(topic, p), timeout=10)
            out.append(TopicPartition(topic, p, high))
        return out
    finally:
        c.close()


def read_from(bootstrap: str, start: list[TopicPartition], camera_id: str, minimum: int):
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": f"reader-{uuid.uuid4().hex}",
                  "enable.auto.commit": False})
    c.assign(start)
    found, deadline = [], time.monotonic() + 20
    try:
        while time.monotonic() < deadline and len(found) < minimum:
            m = c.poll(0.5)
            if m is not None and not m.error() and m.key() == camera_id.encode():
                found.append(m)
        return found
    finally:
        c.close()


def test_worker_publishes_frames_to_video_frames(kafka_bootstrap: str, tmp_path: Path) -> None:
    camera_id = f"test-cam-{uuid.uuid4().hex[:8]}"
    video = generate_traffic_video(tmp_path / "clip.mp4", seconds=VIDEO_SECONDS, fps=SOURCE_FPS,
                                   width=1920, height=1080)
    config = IngestConfig(sample_fps=SAMPLE_FPS, max_width=640, jpeg_quality=70,
                          frames_dir=tmp_path / "frames", retry_initial_seconds=0.1)
    start = end_offsets(kafka_bootstrap, Topic.VIDEO_FRAMES)

    with TypedProducer(kafka_bootstrap, register_shutdown=False) as producer:
        worker = CameraWorker(
            camera_id, VideoSource(SourceKind.FILE, str(video)), config,
            FrameStore(config.frames_dir), lambda m: producer.send(Topic.VIDEO_FRAMES, m),
        )  # fmt: skip
        started = time.monotonic()
        worker.start()
        threading.Event().wait(RUN_SECONDS)
        worker.stop()
        worker.join(timeout=10)
        assert not worker.is_alive()
        elapsed = time.monotonic() - started

    published = worker.metrics.frames_published
    # Real-time pacing: ~SAMPLE_FPS per second, across file loops, not as fast as decoding.
    assert SAMPLE_FPS * RUN_SECONDS * 0.6 <= published <= SAMPLE_FPS * elapsed + 1
    assert worker.metrics.errors == 0
    assert worker.metrics.frames_read > VIDEO_SECONDS * SOURCE_FPS  # looped at least once

    messages = read_from(kafka_bootstrap, start, camera_id, published)
    assert len(messages) == published
    frames = [FrameMessage.model_validate_json(m.value()) for m in messages]
    assert [f.sequence for f in frames] == list(range(published))  # one partition per key
    assert len({f.frame_id for f in frames}) == published
    for f in frames:
        assert f.camera_id == camera_id
        assert (f.width, f.height) == (640, 360)  # 1920x1080 scaled to max_width, 16:9 kept
        assert f.source_fps == pytest.approx(SOURCE_FPS, rel=0.01)
        assert f.encoding == "jpeg"
        path = Path(f.frame_path)
        assert path.parent == config.frames_dir / camera_id
        assert path.read_bytes()[:2] == b"\xff\xd8"
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)
    assert dict(messages[0].headers())["schema"] == b"FrameMessage"


def test_worker_retries_missing_source_without_crashing(tmp_path: Path) -> None:
    config = IngestConfig(frames_dir=tmp_path / "frames", retry_initial_seconds=0.05,
                          retry_max_seconds=0.2)
    published: list[FrameMessage] = []
    worker = CameraWorker("cam-missing", VideoSource(SourceKind.FILE, str(tmp_path / "nope.mp4")),
                          config, FrameStore(config.frames_dir), published.append)
    worker.start()
    time.sleep(1.0)
    worker.stop()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert published == []
    assert worker.metrics.errors >= 3  # kept retrying with backoff
    assert "not found" in (worker.metrics.last_error or "")
    assert worker.metrics.state == "stopped"
