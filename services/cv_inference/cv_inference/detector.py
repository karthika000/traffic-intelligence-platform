"""YOLO detection (PyTorch or OpenVINO backend) and ByteTrack trackers. The only module
that imports Ultralytics/torch."""

import logging
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from cv_inference.analytics import TrackedObject

log = logging.getLogger(__name__)

# COCO class ids for the classes we keep.
COCO_IDS = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
COCO_NAMES = {v: k for k, v in COCO_IDS.items()}


class Detector:
    """One YOLO model instance. Not thread-safe: give each worker thread its own."""

    def __init__(
        self,
        model_path: Path,
        backend: str,
        *,
        imgsz: int,
        threads: int,
        classes: tuple[str, ...],
        min_confidence: float,
    ) -> None:
        self.backend = backend
        self.imgsz = imgsz
        self.class_ids = [COCO_IDS[c] for c in classes]
        self.min_confidence = min_confidence
        if backend == "pytorch":
            torch.set_num_threads(threads)
        self.model = YOLO(str(model_path), task="detect")
        self.detect(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))  # builds the predictor
        if backend == "openvino":
            _tune_openvino(self.model, Path(model_path), threads)

    def detect(self, frame: np.ndarray):
        """Returns Ultralytics ``Boxes`` (numpy) for the kept classes, and wall time in ms."""
        t0 = time.perf_counter()
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.min_confidence,
            classes=self.class_ids,
            verbose=False,
            device="cpu",
        )[0]
        return result.boxes.cpu().numpy(), (time.perf_counter() - t0) * 1000


def _tune_openvino(model: YOLO, export_dir: Path, threads: int) -> None:
    """Recompile the OpenVINO model with a thread cap and without CPU pinning.

    Ultralytics compiles with only PERFORMANCE_HINT=LATENCY, which in the Docker VM spreads
    work over every vCPU and pins threads to cores it can't tell apart (performance vs
    efficiency cores). Measured: 31 ms -> 19 ms per inference with 6 unpinned threads.
    """
    import openvino as ov

    backend = _find_attr_owner(model.predictor.model, "ov_compiled_model")
    if backend is None:
        log.warning("OpenVINO backend object not found; keeping Ultralytics' default compile")
        return
    xml = next(export_dir.glob("*.xml"))
    core = ov.Core()
    backend.ov_compiled_model = core.compile_model(
        core.read_model(str(xml)),
        "CPU",
        {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": threads,
            "ENABLE_CPU_PINNING": False,
        },
    )


def _find_attr_owner(obj: object, attr: str, depth: int = 0) -> object | None:
    if hasattr(obj, attr):
        return obj
    if depth >= 3 or not hasattr(obj, "__dict__"):
        return None
    for value in vars(obj).values():
        if hasattr(value, "__dict__") and not isinstance(value, type):
            found = _find_attr_owner(value, attr, depth + 1)
            if found is not None:
                return found
    return None


def export_openvino(weights: Path, imgsz: int, cache_dir: Path) -> Path:
    """Export (once) to OpenVINO IR; the result is cached under ``cache_dir``."""
    if any(cache_dir.glob("*.xml")):
        return cache_dir
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    out = Path(
        YOLO(str(weights)).export(format="openvino", imgsz=imgsz, dynamic=False, verbose=False)
    )
    out.rename(cache_dir)  # Ultralytics writes <stem>_openvino_model/ next to the weights
    return cache_dir


def benchmark(detector: Detector, frame: np.ndarray, runs: int = 20) -> tuple[float, float]:
    """Mean and p95 latency (ms) of the full predict call, after warm-up."""
    for _ in range(3):
        detector.detect(frame)
    times = sorted(detector.detect(frame)[1] for _ in range(runs))
    return statistics.mean(times), times[round(0.95 * (len(times) - 1))]


def _bytetrack_config() -> dict:
    path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_tracker(lost_seconds: float, fps: float) -> BYTETracker:
    """A ByteTrack instance for one camera. Ultralytics counts track_buffer in frames, so
    convert from seconds at the expected frame rate."""
    cfg = _bytetrack_config()
    cfg["track_buffer"] = max(3, round(lost_seconds * fps))
    return BYTETracker(SimpleNamespace(**cfg))


def track(
    tracker: BYTETracker, boxes, frame: np.ndarray, min_confidence: float
) -> list[TrackedObject]:
    """Update the tracker and return detections with track ids.

    Objects below ``min_confidence`` are kept only if the tracker used them to continue a
    track (ByteTrack's second, low-score association).
    """
    rows = tracker.update(boxes, frame)  # also called with no boxes, so lost tracks age out
    track_by_index = {int(r[7]): int(r[4]) for r in rows}  # x1 y1 x2 y2 id score cls idx
    objects = []
    for i, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls, strict=True)):
        track_id = track_by_index.get(i)
        if conf < min_confidence and track_id is None:
            continue
        label = COCO_NAMES.get(int(cls))
        if label is None:
            continue
        objects.append(TrackedObject(label, float(conf), tuple(float(v) for v in xyxy), track_id))
    return objects
