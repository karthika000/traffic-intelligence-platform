"""Per-track analytics (speed, stopped, in roadway) and per-camera metric windows.

Pure Python: no torch or Ultralytics, so it is unit tested on any machine.
Times are capture timestamps in seconds (frame time, not processing time).
"""

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from cv_inference.zones import ResolvedZone, ground_point
from traffic_common.schemas import TrafficMetrics

VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
RIDDEN_CLASSES = frozenset({"motorcycle", "bicycle"})
PERSON = "person"
RIDER_OVERLAP = 0.3  # share of a person's box covered by a bike for them to count as its rider


@dataclass(frozen=True)
class TrackedObject:
    """One detection in pixel coordinates, with its tracker id if it has one."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    track_id: int | None = None


@dataclass(frozen=True)
class ObjectAnalysis:
    obj: TrackedObject
    speed_kmh: float | None
    stopped: bool
    in_roadway: bool
    rider: bool = False  # a person on a detected bicycle/motorcycle, not a pedestrian

    @property
    def is_vehicle(self) -> bool:
        return self.obj.label in VEHICLE_CLASSES

    @property
    def is_pedestrian_in_roadway(self) -> bool:
        return self.obj.label == PERSON and self.in_roadway and not self.rider


def _overlap_share(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Share of box a's area that lies inside box b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return w * h / area if w > 0 and h > 0 and area > 0 else 0.0


def riders(objects: list[TrackedObject]) -> set[int]:
    """Indices of persons riding a detected bicycle or motorcycle."""
    bikes = [o.bbox for o in objects if o.label in RIDDEN_CLASSES]
    return {
        i
        for i, o in enumerate(objects)
        if o.label == PERSON and any(_overlap_share(o.bbox, b) >= RIDER_OVERLAP for b in bikes)
    }


@dataclass
class _Track:
    label: str
    positions: deque[tuple[float, float, float]] = field(default_factory=deque)  # t, x_m, y_m
    speed_kmh: float | None = None
    slow_since: float | None = None
    last_seen: float = 0.0


class TrackAnalyzer:
    """Speed and stopped state per track for one camera.

    Speed is the straight-line ground displacement between the oldest and newest position
    within ``speed_window_s`` (robust to box jitter), then exponentially smoothed.
    A vehicle is stopped once its speed has stayed below ``stopped_speed_kmh`` for
    ``stopped_after_s``.
    """

    def __init__(
        self,
        speed_window_s: float = 2.0,
        min_span_s: float = 0.4,
        smoothing: float = 0.5,
        stopped_speed_kmh: float = 3.0,
        stopped_after_s: float = 5.0,
        forget_after_s: float = 10.0,
    ) -> None:
        self.speed_window_s = speed_window_s
        self.min_span_s = min_span_s
        self.smoothing = smoothing
        self.stopped_speed_kmh = stopped_speed_kmh
        self.stopped_after_s = stopped_after_s
        self.forget_after_s = forget_after_s
        self.tracks: dict[int, _Track] = {}

    def update(
        self, t: float, objects: list[TrackedObject], zone: ResolvedZone
    ) -> list[ObjectAnalysis]:
        out = []
        rider_indices = riders(objects)
        for i, obj in enumerate(objects):
            gx, gy = ground_point(obj.bbox)
            in_roadway = zone.contains(gx, gy)
            rider = i in rider_indices
            if obj.track_id is None:
                out.append(ObjectAnalysis(obj, None, False, in_roadway, rider))
                continue
            track = self.tracks.setdefault(obj.track_id, _Track(obj.label))
            track.label, track.last_seen = obj.label, t
            speed = self._update_speed(track, t, *zone.to_metres(gx, gy))
            stopped = (
                self._update_stopped(track, t, speed) if obj.label in VEHICLE_CLASSES else False
            )
            out.append(ObjectAnalysis(obj, speed, stopped, in_roadway, rider))
        self._forget(t)
        return out

    def _update_speed(self, track: _Track, t: float, x: float, y: float) -> float | None:
        track.positions.append((t, x, y))
        while track.positions and track.positions[0][0] < t - self.speed_window_s:
            track.positions.popleft()
        t0, x0, y0 = track.positions[0]
        span = t - t0
        if span >= self.min_span_s:
            raw = math.hypot(x - x0, y - y0) / span * 3.6
            track.speed_kmh = (
                raw
                if track.speed_kmh is None
                else (self.smoothing * raw + (1 - self.smoothing) * track.speed_kmh)
            )
        return track.speed_kmh

    def _update_stopped(self, track: _Track, t: float, speed: float | None) -> bool:
        if speed is None or speed >= self.stopped_speed_kmh:
            track.slow_since = None
            return False
        if track.slow_since is None:
            track.slow_since = t
        return t - track.slow_since >= self.stopped_after_s

    def _forget(self, t: float) -> None:
        for track_id in [
            k for k, v in self.tracks.items() if t - v.last_seen > self.forget_after_s
        ]:
            del self.tracks[track_id]


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy's default), q in [0, 100]."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class _Window:
    start: float
    frames: int = 0
    tracks_by_class: dict[str, set[int]] = field(default_factory=dict)
    vehicle_speeds: dict[int, list[float]] = field(default_factory=dict)
    vehicles_in_roadway: list[int] = field(default_factory=list)
    stopped: set[int] = field(default_factory=set)
    pedestrian_tracks_in_roadway: set[int] = field(default_factory=set)
    max_untracked_pedestrians_in_roadway: int = 0
    confidences: list[float] = field(default_factory=list)


class MetricsAggregator:
    """Rolls per-frame analyses into fixed TrafficMetrics windows aligned to the clock
    (e.g. :00, :05, :10 for 5 s windows), one camera per aggregator."""

    def __init__(self, camera_id: str, window_s: float = 5.0, model_version: str | None = None):
        self.camera_id = camera_id
        self.window_s = window_s
        self.model_version = model_version
        self._window: _Window | None = None

    def add(self, t: float, analyses: list[ObjectAnalysis]) -> TrafficMetrics | None:
        """Add one frame. Returns the previous window's metrics if this frame starts a new one."""
        start = math.floor(t / self.window_s) * self.window_s
        closed = None
        if self._window is not None and start != self._window.start:
            closed = self._close()
        if self._window is None:
            self._window = _Window(start)
        self._accumulate(self._window, analyses)
        return closed

    def poll(self, now: float, grace_s: float = 2.0) -> TrafficMetrics | None:
        """Close the open window if its end has passed (no frames arrived to close it)."""
        if self._window is not None and now >= self._window.start + self.window_s + grace_s:
            return self._close()
        return None

    def _accumulate(self, w: _Window, analyses: list[ObjectAnalysis]) -> None:
        w.frames += 1
        in_roadway = untracked_peds = 0
        for a in analyses:
            w.confidences.append(a.obj.confidence)
            tid = a.obj.track_id
            if a.is_vehicle and a.in_roadway:
                in_roadway += 1
            if a.is_pedestrian_in_roadway:
                if tid is None:
                    untracked_peds += 1
                else:
                    w.pedestrian_tracks_in_roadway.add(tid)
            if tid is None:
                continue
            w.tracks_by_class.setdefault(a.obj.label, set()).add(tid)
            if a.is_vehicle:
                if a.speed_kmh is not None:
                    w.vehicle_speeds.setdefault(tid, []).append(a.speed_kmh)
                if a.stopped and a.in_roadway:  # parked cars off the carriageway aren't incidents
                    w.stopped.add(tid)
        w.vehicles_in_roadway.append(in_roadway)
        w.max_untracked_pedestrians_in_roadway = max(
            w.max_untracked_pedestrians_in_roadway, untracked_peds
        )

    def _close(self) -> TrafficMetrics:
        w = self._window
        assert w is not None
        self._window = None
        per_track = [sum(v) / len(v) for v in w.vehicle_speeds.values()]
        vehicles = set().union(
            *(ids for label, ids in w.tracks_by_class.items() if label in VEHICLE_CLASSES)
        )
        return TrafficMetrics(
            camera_id=self.camera_id,
            window_start=datetime.fromtimestamp(w.start, UTC),
            window_end=datetime.fromtimestamp(w.start + self.window_s, UTC),
            timestamp=datetime.fromtimestamp(w.start + self.window_s, UTC),
            vehicle_count=len(vehicles),
            counts_by_class={label: len(ids) for label, ids in sorted(w.tracks_by_class.items())},
            avg_speed_kmh=round(sum(per_track) / len(per_track), 1) if per_track else None,
            p85_speed_kmh=round(percentile(per_track, 85), 1) if per_track else None,
            density=round(sum(w.vehicles_in_roadway) / len(w.vehicles_in_roadway), 2),
            stopped_vehicles=len(w.stopped),
            pedestrians_in_roadway=len(w.pedestrian_tracks_in_roadway)
            + w.max_untracked_pedestrians_in_roadway,
            mean_confidence=round(sum(w.confidences) / len(w.confidences), 3)
            if w.confidences
            else None,
            frames=w.frames,
            model_version=self.model_version,
        )
