"""Integration tests: the real YOLO detector + ByteTrack on frames from a sample video.

Needs Ultralytics/torch (installed in the Docker test image; skipped elsewhere) and
data/videos/cam-hyde-park-corner.webm (scripts/download_sample_videos.py). Weights come
from the cv_inference weights volume when mounted at /weights, else are downloaded.
The Kafka test publishes to the real `detections` topic under a `test-cam-...` id.
"""

import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from conftest import SMART_APP_CONTROL_HINT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "cv_inference"))
VIDEO = ROOT / "data" / "videos" / "cam-hyde-park-corner.webm"

pytest.importorskip(
    "ultralytics", reason="Ultralytics not installed (runs in the Docker test image)"
)
import cv2  # noqa: E402

from cv_inference.analytics import MetricsAggregator, TrackAnalyzer  # noqa: E402
from cv_inference.config import CvConfig  # noqa: E402
from cv_inference.detector import Detector, make_tracker, track  # noqa: E402
from cv_inference.pipeline import CameraContext, process_frame  # noqa: E402
from cv_inference.registry import pretrained_weights  # noqa: E402
from cv_inference.zones import ZoneConfig  # noqa: E402
from traffic_common.schemas import Detection, FrameMessage  # noqa: E402
from traffic_common.topics import Topic  # noqa: E402

FPS = 5
N_FRAMES = 12


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cached = Path("/weights/pretrained/yolo26n.pt")
    if cached.exists():
        return cached
    return pretrained_weights("yolo26n.pt", tmp_path_factory.mktemp("weights"))


@pytest.fixture(scope="module")
def video_frames(tmp_path_factory: pytest.TempPathFactory) -> list[tuple[FrameMessage, object]]:
    """N_FRAMES frames sampled at 5 fps, written as JPEGs like video_ingest does."""
    if not VIDEO.exists():
        pytest.skip(f"{VIDEO.name} missing; run scripts/download_sample_videos.py")
    out = tmp_path_factory.mktemp("frames")
    cap = cv2.VideoCapture(str(VIDEO))
    step = round(cap.get(cv2.CAP_PROP_FPS) / FPS)
    camera_id = f"test-cam-{uuid.uuid4().hex[:8]}"
    start = datetime.now(UTC) - timedelta(seconds=N_FRAMES / FPS)
    frames, i = [], 0
    while len(frames) < N_FRAMES:
        ok, image = cap.read()
        if not ok:
            break
        if i % step == 0:
            path = out / f"{len(frames):04d}.jpg"
            cv2.imwrite(str(path), image)
            msg = FrameMessage(
                camera_id=camera_id,
                sequence=len(frames),
                frame_path=str(path),
                width=image.shape[1],
                height=image.shape[0],
                source_fps=30,
                timestamp=start + timedelta(seconds=len(frames) / FPS),
            )
            frames.append((msg, image))
        i += 1
    cap.release()
    assert len(frames) == N_FRAMES
    return frames


def run_pipeline(weights: Path, frames) -> tuple[list[Detection], list]:
    cfg = CvConfig(threads=4)
    detector = Detector(
        weights,
        "pytorch",
        imgsz=cfg.imgsz,
        threads=cfg.threads,
        classes=cfg.classes,
        min_confidence=cfg.tracker_confidence,
    )
    tracker = make_tracker(cfg.track_lost_seconds, FPS)
    camera_id = frames[0][0].camera_id
    ctx = CameraContext(
        camera_id,
        lambda b, img: track(tracker, b, img, cfg.confidence),
        TrackAnalyzer(),
        MetricsAggregator(camera_id, window_s=1.0, model_version="vehicle-detector/test"),
        zone_config=ZoneConfig(
            roadway=[(0, 0.40), (1, 0.47), (1, 0.96), (0, 0.96)],
            pixels_per_metre=30,
            reference_width=854,
        ),
    )
    detections, metrics = [], []
    for msg, image in frames:
        result = process_frame(
            ctx,
            msg,
            image,
            detector.detect,
            model_name="vehicle-detector",
            model_version="test",
            default_frame_width_metres=30,
        )
        detections.append(result.detection)
        if result.metrics:
            metrics.append(result.metrics)
    return detections, metrics


def test_detector_produces_valid_tracked_detections(weights: Path, video_frames) -> None:
    detections, metrics = run_pipeline(weights, video_frames)

    assert len(detections) == N_FRAMES
    for det, (msg, _) in zip(detections, video_frames, strict=True):
        Detection.model_validate_json(det.model_dump_json())  # survives the wire format
        assert det.frame_id == msg.frame_id and det.camera_id == msg.camera_id
        assert det.inference_ms and det.inference_ms > 0
    objects = [o for d in detections for o in d.objects]
    labels = {o.label for o in objects}
    assert "car" in labels  # the clip has several cars in every frame
    assert labels <= {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
    assert all(o.confidence >= 0.1 for o in objects)

    # ByteTrack keeps identities across frames at 5 fps.
    frames_per_track: dict[int, int] = {}
    for o in objects:
        if o.track_id is not None:
            frames_per_track[o.track_id] = frames_per_track.get(o.track_id, 0) + 1
    assert max(frames_per_track.values()) >= N_FRAMES // 2
    speeds = [o.speed_kmh for o in objects if o.speed_kmh is not None]
    assert speeds and all(0 <= s < 200 for s in speeds)
    assert any(o.in_roadway for o in objects)

    assert metrics, "1 s windows over 2.4 s of frames should close at least one window"
    assert metrics[0].frames >= 1 and metrics[0].vehicle_count >= 1
    assert metrics[0].model_version == "vehicle-detector/test"


def test_detections_round_trip_through_kafka(
    kafka_bootstrap: str, weights: Path, video_frames
) -> None:
    pytest.importorskip("confluent_kafka", reason=SMART_APP_CONTROL_HINT, exc_type=ImportError)
    from confluent_kafka import Consumer, TopicPartition

    from traffic_common.messaging import TypedProducer

    detections, _ = run_pipeline(weights, video_frames[:4])
    camera_id = detections[0].camera_id
    probe = Consumer(
        {"bootstrap.servers": kafka_bootstrap, "group.id": f"probe-{uuid.uuid4().hex}"}
    )
    parts = probe.list_topics(Topic.DETECTIONS, timeout=10).topics[Topic.DETECTIONS].partitions
    start = [
        TopicPartition(
            Topic.DETECTIONS,
            p,
            probe.get_watermark_offsets(TopicPartition(Topic.DETECTIONS, p), timeout=10)[1],
        )
        for p in parts
    ]
    probe.close()

    with TypedProducer(kafka_bootstrap, register_shutdown=False) as producer:
        for det in detections:
            producer.send(Topic.DETECTIONS, det)

    reader = Consumer(
        {
            "bootstrap.servers": kafka_bootstrap,
            "group.id": f"reader-{uuid.uuid4().hex}",
            "enable.auto.commit": False,
        }
    )
    reader.assign(start)
    got, deadline = [], time.monotonic() + 20
    while len(got) < len(detections) and time.monotonic() < deadline:
        m = reader.poll(0.5)
        if m is not None and not m.error() and m.key() == camera_id.encode():
            got.append(Detection.model_validate_json(m.value()))
    reader.close()
    assert [d.frame_id for d in got] == [d.frame_id for d in detections]
    assert got[0].objects == detections[0].objects
