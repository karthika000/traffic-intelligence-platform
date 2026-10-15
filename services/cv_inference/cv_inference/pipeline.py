"""Frame scheduling, per-camera state, message building and stats. No Ultralytics imports:
the detector and tracker are injected, so this is unit tested without torch."""

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import cv2
import numpy as np

from cv_inference.analytics import (
    MetricsAggregator,
    ObjectAnalysis,
    TrackAnalyzer,
    TrackedObject,
    percentile,
)
from cv_inference.zones import ResolvedZone, ZoneConfig
from traffic_common.schemas import (
    BoundingBox,
    DetectedObject,
    Detection,
    FrameMessage,
    TrafficMetrics,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- scheduling


class LatestFrames:
    """Holds at most one pending frame per camera; a newer frame replaces the waiting one.

    Cameras are served round-robin, and a camera is never handed to two workers at once
    (its tracker must see frames in order).
    """

    def __init__(self) -> None:
        self._pending: dict[str, FrameMessage] = {}
        self._ready: deque[str] = deque()
        self._busy: set[str] = set()
        self._cond = threading.Condition()

    def put(self, frame: FrameMessage) -> FrameMessage | None:
        """Queue a frame; returns the older frame it replaced, if any."""
        with self._cond:
            replaced = self._pending.get(frame.camera_id)
            self._pending[frame.camera_id] = frame
            if replaced is None and frame.camera_id not in self._busy:
                self._ready.append(frame.camera_id)
                self._cond.notify()
            return replaced

    def take(self, timeout: float) -> FrameMessage | None:
        with self._cond:
            if not self._ready and not self._cond.wait_for(lambda: bool(self._ready), timeout):
                return None
            camera_id = self._ready.popleft()
            self._busy.add(camera_id)
            return self._pending.pop(camera_id)

    def done(self, camera_id: str) -> None:
        with self._cond:
            self._busy.discard(camera_id)
            if camera_id in self._pending:
                self._ready.append(camera_id)
                self._cond.notify()

    def pending(self) -> int:
        with self._cond:
            return len(self._pending)


# --------------------------------------------------------------------------- stats


@dataclass
class CameraStats:
    processed: int = 0
    stale: int = 0
    superseded: int = 0
    missing: int = 0
    errors: int = 0
    inference_ms: list[float] = field(default_factory=list)
    lag_ms: list[float] = field(default_factory=list)  # capture -> Detection published

    def snapshot_and_reset(self, interval_s: float) -> str:
        inf, lag = self.inference_ms, self.lag_ms
        text = (
            f"processed={self.processed} fps={self.processed / interval_s:.2f} "
            f"inference_ms mean={_mean(inf)} p95={_p95(inf)} lag_ms mean={_mean(lag)} "
            f"stale={self.stale} superseded={self.superseded} missing={self.missing} "
            f"errors={self.errors}"
        )
        self.__init__()  # type: ignore[misc]
        return text


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.1f}" if values else "-"


def _p95(values: list[float]) -> str:
    return f"{percentile(values, 95):.1f}" if values else "-"


# --------------------------------------------------------------------------- per camera


class Tracker(Protocol):
    def __call__(self, boxes: Any, frame: np.ndarray) -> list[TrackedObject]: ...


@dataclass
class CameraContext:
    camera_id: str
    tracker: Tracker  # boxes, frame -> tracked objects
    analyzer: TrackAnalyzer
    aggregator: MetricsAggregator
    zone_config: ZoneConfig = field(default_factory=ZoneConfig)
    stats: CameraStats = field(default_factory=CameraStats)
    lock: threading.Lock = field(default_factory=threading.Lock)
    _resolved: tuple[tuple[int, int], ResolvedZone] | None = None

    def zone(self, width: int, height: int, default_frame_width_metres: float) -> ResolvedZone:
        if self._resolved is None or self._resolved[0] != (width, height):
            self._resolved = (
                (width, height),
                self.zone_config.resolve(width, height, default_frame_width_metres),
            )
        return self._resolved[1]

    def set_zone_config(self, config: ZoneConfig) -> None:
        if config != self.zone_config:
            self.zone_config, self._resolved = config, None


# --------------------------------------------------------------------------- messages


def to_detected_object(a: ObjectAnalysis, width: int, height: int) -> DetectedObject | None:
    """Pixel box -> normalised DetectedObject; None if the box is degenerate after clipping."""
    x1, y1, x2, y2 = a.obj.bbox
    nx1, nx2 = max(0.0, x1 / width), min(1.0, x2 / width)
    ny1, ny2 = max(0.0, y1 / height), min(1.0, y2 / height)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return DetectedObject(
        label=a.obj.label,
        confidence=round(min(1.0, max(0.0, a.obj.confidence)), 4),
        bbox=BoundingBox(
            x_min=round(nx1, 5), y_min=round(ny1, 5), x_max=round(nx2, 5), y_max=round(ny2, 5)
        ),
        track_id=a.obj.track_id,
        speed_kmh=None if a.speed_kmh is None else round(a.speed_kmh, 1),
        stopped=a.stopped,
        in_roadway=a.in_roadway,
    )


def build_detection(
    frame: FrameMessage,
    analyses: list[ObjectAnalysis],
    width: int,
    height: int,
    model_name: str,
    model_version: str,
    inference_ms: float,
) -> Detection:
    objects = [o for a in analyses if (o := to_detected_object(a, width, height)) is not None]
    return Detection(
        camera_id=frame.camera_id,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        model_name=model_name,
        model_version=model_version,
        objects=objects,
        inference_ms=round(inference_ms, 2),
    )


def read_frame(path: str) -> np.ndarray | None:
    """Decode a frame file; None if it no longer exists (retention already removed it)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except FileNotFoundError:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {path}")
    return image


@dataclass
class FrameResult:
    detection: Detection
    metrics: TrafficMetrics | None
    analyses: list[ObjectAnalysis]
    zone: ResolvedZone


def process_frame(
    ctx: CameraContext,
    frame_msg: FrameMessage,
    image: np.ndarray,
    detect: Callable[[np.ndarray], tuple[Any, float]],
    *,
    model_name: str,
    model_version: str,
    default_frame_width_metres: float,
) -> FrameResult:
    """Detect, track and analyse one frame. Call with ``ctx.lock`` held."""
    height, width = image.shape[:2]
    boxes, inference_ms = detect(image)
    objects = ctx.tracker(boxes, image)
    zone = ctx.zone(width, height, default_frame_width_metres)
    t = frame_msg.timestamp.timestamp()
    analyses = ctx.analyzer.update(t, objects, zone)
    metrics = ctx.aggregator.add(t, analyses)
    detection = build_detection(
        frame_msg, analyses, width, height, model_name, model_version, inference_ms
    )
    ctx.stats.inference_ms.append(inference_ms)
    return FrameResult(detection, metrics, analyses, zone)


def frame_age_seconds(frame: FrameMessage, now: datetime | None = None) -> float:
    return ((now or datetime.now(UTC)) - frame.timestamp).total_seconds()
