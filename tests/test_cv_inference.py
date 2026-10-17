"""Unit tests for cv_inference analytics (no torch, Ultralytics, Kafka or Postgres needed)."""

import importlib.util
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "cv_inference"))

pytest.importorskip("cv2")
from cv_inference.analytics import (  # noqa: E402
    MetricsAggregator,
    ObjectAnalysis,
    TrackAnalyzer,
    TrackedObject,
    percentile,
)
from cv_inference.config import CvConfig, default_threads  # noqa: E402
from cv_inference.pipeline import (  # noqa: E402
    CameraContext,
    LatestFrames,
    build_detection,
    frame_age_seconds,
    read_frame,
    to_detected_object,
)
from cv_inference.zones import ResolvedZone, ZoneConfig, ground_point  # noqa: E402
from traffic_common.schemas import FrameMessage  # noqa: E402

W, H = 1000, 500
SQUARE = ResolvedZone([(0, 0), (100, 0), (100, 100), (0, 100)], ppm_x=10, ppm_y=10)


def box_at(x: float, y: float, w: float = 40, h: float = 20) -> tuple[float, float, float, float]:
    """A box whose ground point (bottom centre) is at (x, y)."""
    return (x - w / 2, y - h, x + w / 2, y)


def car(track_id: int | None, x: float, y: float = 50, label: str = "car", conf: float = 0.9):
    return TrackedObject(label, conf, box_at(x, y), track_id)


# ----------------------------------------------------------------------------- zones


def test_point_in_polygon() -> None:
    triangle = ResolvedZone([(0, 0), (10, 0), (0, 10)], 1, 1)
    assert triangle.contains(2, 2)
    assert not triangle.contains(8, 8)
    assert not triangle.contains(-1, 1)
    concave = ResolvedZone([(0, 0), (10, 0), (10, 10), (5, 3), (0, 10)], 1, 1)
    assert concave.contains(1, 5) and concave.contains(9, 5)
    assert not concave.contains(5, 8)  # inside the notch


def test_zone_config_scales_to_frame_size() -> None:
    cfg = ZoneConfig(
        roadway=[(0, 0.5), (1, 0.5), (1, 1), (0, 1)],
        pixels_per_metre=40,
        pixels_per_metre_y=10,
        reference_width=854,
    )
    zone = cfg.resolve(1708, 960, default_frame_width_metres=30)
    assert zone.polygon == [(0, 480), (1708, 480), (1708, 960), (0, 960)]
    assert (zone.ppm_x, zone.ppm_y) == (80, 20)  # double the resolution, double the px/m
    assert zone.contains(800, 700) and not zone.contains(800, 100)
    assert not zone.whole_frame and not zone.default_scale


def test_missing_config_uses_whole_frame_and_default_scale() -> None:
    zone = ZoneConfig().resolve(W, H, default_frame_width_metres=40)
    assert zone.whole_frame and zone.default_scale
    assert zone.contains(1, 1) and zone.contains(W - 1, H - 1)
    assert zone.ppm_x == zone.ppm_y == 25  # 1000 px / 40 m


def test_zone_config_parse_is_lenient() -> None:
    legacy = {"zones": [{"name": "x", "polygon": [[0, 0], [1, 0], [1, 1]]}], "count_lines": []}
    config, problem = ZoneConfig.parse(legacy)
    assert problem is None and config.roadway is None  # unknown keys ignored
    config, problem = ZoneConfig.parse({"roadway": [[0, 0], [2, 0], [1, 1]]})
    assert config == ZoneConfig() and "normalised" in problem
    config, problem = ZoneConfig.parse(None)
    assert config == ZoneConfig() and problem is None


def test_ground_point_is_bottom_centre() -> None:
    assert ground_point((10, 20, 30, 60)) == (20, 60)


def test_demo_zone_configs_are_valid() -> None:
    spec = importlib.util.spec_from_file_location("wzc", ROOT / "scripts" / "write_zone_configs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.DEMO_ZONE_CONFIGS) == {
        "cam-hyde-park-corner",
        "cam-trafalgar-square",
        "cam-elephant-castle",
        "cam-old-street",
    }
    for raw in module.DEMO_ZONE_CONFIGS.values():
        cfg = ZoneConfig.model_validate(raw)
        assert cfg.roadway and cfg.pixels_per_metre
        zone = cfg.resolve(854, 480, 30)
        assert zone.contains(427, 432)  # lower centre of these road scenes is roadway


# ----------------------------------------------------------------------------- speed


def drive(
    analyzer: TrackAnalyzer,
    speed_mps: float,
    seconds: float,
    fps: float = 5,
    zone: ResolvedZone = SQUARE,
    start_x: float = 0,
    track_id: int = 1,
    t0: float = 0.0,
):
    """Move a car along x at a constant speed; returns the analysis per frame."""
    out = []
    for i in range(int(seconds * fps) + 1):
        t = t0 + i / fps
        x = start_x + speed_mps * (t - t0) * zone.ppm_x
        out.append(analyzer.update(t, [car(track_id, x)], zone)[0])
    return out


def test_constant_speed_is_measured() -> None:
    results = drive(TrackAnalyzer(), speed_mps=10, seconds=4)
    assert results[0].speed_kmh is None  # one position: no speed yet
    assert results[-1].speed_kmh == pytest.approx(36.0, rel=0.01)


def test_speed_uses_separate_vertical_scale() -> None:
    zone = ResolvedZone(SQUARE.polygon, ppm_x=40, ppm_y=10)
    analyzer = TrackAnalyzer()
    for i in range(11):  # 5 m/s towards the camera = 50 px/s vertically at 10 px/m
        a = analyzer.update(i / 5, [car(1, 50, 10 + 10 * i)], zone)[0]
    assert a.speed_kmh == pytest.approx(18.0, rel=0.01)


def test_speed_is_smoothed_against_jitter() -> None:
    analyzer = TrackAnalyzer(smoothing=0.5)
    jitter = [0, 3, -3, 3, -3, 0, 3, -3, 3, -3, 0, 2]
    speeds = [
        analyzer.update(i / 5, [car(1, 20 + j)], SQUARE)[0].speed_kmh for i, j in enumerate(jitter)
    ]  # a parked car whose box wobbles by +-3 px (0.3 m)
    assert max(s for s in speeds if s is not None) < 3.0


def test_speed_window_forgets_old_positions() -> None:
    analyzer = TrackAnalyzer(speed_window_s=2.0, smoothing=1.0)
    drive(analyzer, speed_mps=20, seconds=3)  # fast...
    slow = drive(analyzer, speed_mps=2, seconds=3, start_x=60 * 10, t0=3.2)  # ...then slow
    assert slow[-1].speed_kmh == pytest.approx(7.2, rel=0.05)


# ----------------------------------------------------------------------------- stopped


def test_stopped_after_threshold_duration() -> None:
    analyzer = TrackAnalyzer(stopped_speed_kmh=3, stopped_after_s=5)
    results = [analyzer.update(i / 5, [car(7, 50)], SQUARE)[0] for i in range(51)]  # 10 s parked
    first_slow = next(i for i, r in enumerate(results) if r.speed_kmh is not None)
    stopped_at = next(i for i, r in enumerate(results) if r.stopped)
    assert (stopped_at - first_slow) / 5 == pytest.approx(5.0, abs=0.21)
    assert all(r.stopped for r in results[stopped_at:])


def test_stopped_resets_when_vehicle_moves() -> None:
    analyzer = TrackAnalyzer(stopped_speed_kmh=3, stopped_after_s=2, smoothing=1.0)
    for i in range(25):
        a = analyzer.update(i / 5, [car(1, 50)], SQUARE)[0]
    assert a.stopped
    for i in range(25, 40):  # drives off at 5 m/s
        a = analyzer.update(i / 5, [car(1, 50 + (i - 24) * 10)], SQUARE)[0]
    assert not a.stopped and a.speed_kmh > 10


def test_people_are_never_stopped_vehicles() -> None:
    analyzer = TrackAnalyzer(stopped_after_s=1)
    for i in range(20):
        a = analyzer.update(i / 5, [car(3, 50, label="person")], SQUARE)[0]
    assert not a.stopped and a.speed_kmh == pytest.approx(0, abs=0.5)


def test_in_roadway_and_untracked_objects() -> None:
    analyzer = TrackAnalyzer()
    inside, outside, untracked = analyzer.update(
        0.0, [car(1, 50, 50, "person"), car(2, 500, 50, "person"), car(None, 50, 50)], SQUARE
    )
    assert inside.in_roadway and not outside.in_roadway
    assert untracked.speed_kmh is None and untracked.in_roadway and not untracked.stopped


def test_riders_are_not_pedestrians() -> None:
    analyzer = TrackAnalyzer()
    bike = TrackedObject("motorcycle", 0.9, (40, 40, 60, 90), 1)
    rider = TrackedObject("person", 0.9, (42, 20, 58, 80), 2)  # sits on the bike
    walker = TrackedObject("person", 0.9, (10, 40, 20, 90), 3)  # next to it, no overlap
    a_bike, a_rider, a_walker = analyzer.update(0.0, [bike, rider, walker], SQUARE)
    assert a_rider.rider and a_rider.in_roadway and not a_rider.is_pedestrian_in_roadway
    assert not a_walker.rider and a_walker.is_pedestrian_in_roadway
    assert not a_bike.rider

    agg = MetricsAggregator("cam", window_s=5)
    agg.add(0.0, [a_bike, a_rider, a_walker])
    m = agg.poll(10.0)
    assert m.pedestrians_in_roadway == 1  # the walker only


def test_old_tracks_are_forgotten() -> None:
    analyzer = TrackAnalyzer(forget_after_s=10)
    analyzer.update(0.0, [car(1, 10)], SQUARE)
    analyzer.update(5.0, [car(2, 10)], SQUARE)
    analyzer.update(12.0, [car(3, 10)], SQUARE)
    assert set(analyzer.tracks) == {2, 3}


# ----------------------------------------------------------------------------- metrics


def analysis(tid, label="car", speed=None, stopped=False, in_roadway=True, conf=0.8):
    return ObjectAnalysis(
        TrackedObject(label, conf, (0, 0, 10, 10), tid), speed, stopped, in_roadway
    )


def test_percentile_matches_numpy_definition() -> None:
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 85) == pytest.approx(44.0)
    assert percentile(values, 50) == 30 and percentile([7.0], 85) == 7.0


def test_metrics_window_aggregation() -> None:
    agg = MetricsAggregator("cam-a", window_s=5.0, model_version="vehicle-detector/1")
    base = 1_800_000_000.0  # aligned to a 5 s boundary
    frames = [
        [
            analysis(1, speed=30),
            analysis(2, "truck", speed=10),
            analysis(9, "person", in_roadway=True),
            analysis(None, "person", in_roadway=True),
        ],
        [
            analysis(1, speed=34),
            analysis(2, "truck", speed=0.5, stopped=True),
            analysis(3, "bus", in_roadway=False),
        ],
        [
            analysis(1, speed=32),
            analysis(4, "bicycle", speed=15, conf=0.4),
            analysis(5, speed=0.2, stopped=True, in_roadway=False),
        ],  # parked off the road
    ]
    for i, f in enumerate(frames):
        assert agg.add(base + i, f) is None
    m = agg.add(base + 5.2, [analysis(1, speed=31)])  # next window closes the first
    assert m is not None
    assert m.window_start == datetime.fromtimestamp(base, UTC)
    assert m.window_end - m.window_start == timedelta(seconds=5)
    assert m.counts_by_class == {"bicycle": 1, "bus": 1, "car": 2, "person": 1, "truck": 1}
    assert m.vehicle_count == 5  # cars 1 and 5, truck 2, bus 3, bicycle 4 (person excluded)
    per_track = [32.0, 5.25, 15.0, 0.2]  # mean speed per vehicle track; bus has none
    assert m.avg_speed_kmh == pytest.approx(round(sum(per_track) / len(per_track), 1))
    assert m.p85_speed_kmh == pytest.approx(round(percentile(per_track, 85), 1))
    assert m.density == pytest.approx((2 + 2 + 2) / 3)  # vehicles inside the roadway per frame
    assert m.stopped_vehicles == 1  # truck 2; car 5 is stopped but parked off the roadway
    assert m.pedestrians_in_roadway == 2  # tracked person 9 + one untracked
    assert m.frames == 3 and m.model_version == "vehicle-detector/1"
    assert m.mean_confidence == pytest.approx(round((0.8 * 9 + 0.4) / 10, 3))


def test_empty_window_and_poll() -> None:
    agg = MetricsAggregator("cam-a", window_s=5.0)
    assert agg.poll(1_000_000.0) is None  # nothing open
    agg.add(1_000_000.0, [])
    assert agg.poll(1_000_005.0) is None  # within grace
    m = agg.poll(1_000_008.0)
    assert m is not None and m.frames == 1 and m.vehicle_count == 0
    assert m.avg_speed_kmh is None and m.p85_speed_kmh is None and m.mean_confidence is None
    assert agg.poll(1_000_020.0) is None  # closed exactly once


# ----------------------------------------------------------------------------- scheduling


def frame_msg(camera_id: str, seq: int, age_s: float = 0.0) -> FrameMessage:
    return FrameMessage(
        camera_id=camera_id,
        sequence=seq,
        frame_path=f"/x/{camera_id}/{seq}.jpg",
        width=W,
        height=H,
        source_fps=25,
        timestamp=datetime.now(UTC) - timedelta(seconds=age_s),
    )


def test_latest_frames_keeps_only_newest_per_camera() -> None:
    slots = LatestFrames()
    assert slots.put(frame_msg("a", 1)) is None
    assert slots.put(frame_msg("a", 2)).sequence == 1  # replaced while waiting
    assert slots.put(frame_msg("b", 1)) is None
    first = slots.take(0.1)
    assert (first.camera_id, first.sequence) == ("a", 2)
    assert slots.put(frame_msg("a", 3)) is None  # a is busy: wait, don't queue twice
    second = slots.take(0.1)
    assert second.camera_id == "b"
    assert slots.take(0.05) is None  # a's new frame waits until a is done
    slots.done("a")
    assert slots.take(0.1).sequence == 3


def test_latest_frames_round_robin_and_wakeup() -> None:
    slots = LatestFrames()
    got = []
    t = threading.Thread(target=lambda: got.append(slots.take(2.0)))
    t.start()
    slots.put(frame_msg("a", 1))
    t.join(2)
    assert got[0].camera_id == "a"


def test_frame_age() -> None:
    assert frame_age_seconds(frame_msg("a", 1, age_s=7)) == pytest.approx(7, abs=0.5)


def test_read_frame_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_frame(str(tmp_path / "gone.jpg")) is None
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    with pytest.raises(ValueError):
        read_frame(str(bad))


# ----------------------------------------------------------------------------- messages


def test_detected_object_normalises_and_clips() -> None:
    a = ObjectAnalysis(TrackedObject("car", 0.87654, (-10, 100, 500, 520), 4), 23.456, True, True)
    obj = to_detected_object(a, W, H)
    assert (obj.bbox.x_min, obj.bbox.y_min, obj.bbox.x_max, obj.bbox.y_max) == (0.0, 0.2, 0.5, 1.0)
    assert obj.speed_kmh == 23.5 and obj.stopped and obj.in_roadway and obj.track_id == 4
    assert obj.confidence == 0.8765
    off_frame = ObjectAnalysis(TrackedObject("car", 0.9, (1100, 10, 1200, 50)), None, False, False)
    assert to_detected_object(off_frame, W, H) is None


def test_build_detection_keeps_frame_identity() -> None:
    frame = frame_msg("cam-a", 12)
    det = build_detection(frame, [analysis(1, speed=20)], W, H, "vehicle-detector", "1", 17.333)
    assert det.frame_id == frame.frame_id and det.timestamp == frame.timestamp
    assert det.camera_id == "cam-a" and det.inference_ms == 17.33
    assert det.model_name == "vehicle-detector" and det.model_version == "1"
    assert len(det.objects) == 1 and det.objects[0].speed_kmh == 20


def test_camera_context_resolves_zone_per_size() -> None:
    ctx = CameraContext(
        "a", tracker=lambda b, f: [], analyzer=TrackAnalyzer(), aggregator=MetricsAggregator("a")
    )
    z1 = ctx.zone(W, H, 25)
    assert ctx.zone(W, H, 25) is z1  # cached
    ctx.set_zone_config(ZoneConfig(pixels_per_metre=50))
    assert ctx.zone(W, H, 25).ppm_x == 50


# ----------------------------------------------------------------------------- config


def test_config_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = CvConfig()
    assert cfg.model == "yolo26n.pt" and cfg.imgsz == 480 and cfg.backend == "openvino"
    assert cfg.threads == default_threads() and 1 <= cfg.threads <= 6
    monkeypatch.setenv("CV_BACKEND", "auto")
    monkeypatch.setenv("CV_THREADS", "4")
    monkeypatch.setenv("CV_ANNOTATE", "false")
    monkeypatch.setenv("FRAME_SAMPLE_FPS", "5")
    env = CvConfig.from_env()
    assert (env.backend, env.threads, env.annotate, env.expected_fps) == ("auto", 4, False, 5.0)
    with pytest.raises(ValueError):
        CvConfig(backend="cuda")


def test_frame_clock_uses_the_display_zone() -> None:
    from zoneinfo import ZoneInfo

    from cv_inference.annotate import frame_clock

    assert frame_clock(datetime(2026, 10, 17, 10, 17, 44, tzinfo=UTC)) == "11:17:44 BST"
    assert frame_clock(datetime(2026, 12, 1, 10, 17, 44)) == "10:17:44 GMT"  # naive = UTC
    assert frame_clock(datetime(2026, 10, 17, 10, 17, 44, tzinfo=UTC), ZoneInfo("UTC")) == (
        "10:17:44 UTC"
    )
