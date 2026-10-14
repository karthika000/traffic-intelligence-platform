"""Unit tests for video_ingest: sampling, resizing, source resolution, storage and retention."""

import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "video_ingest"))

cv2 = pytest.importorskip("cv2", exc_type=ImportError)
import numpy as np  # noqa: E402

from video_ingest.config import IngestConfig  # noqa: E402
from video_ingest.metrics import CameraMetrics  # noqa: E402
from video_ingest.sampling import FrameSampler, encode_jpeg, resize_to_max_width  # noqa: E402
from video_ingest.sources import SourceKind, resolve_source  # noqa: E402
from video_ingest.storage import (  # noqa: E402
    FrameStore,
    cleanup_expired_frames,
    frame_filename,
    parse_frame_timestamp,
)

# ----------------------------------------------------------------------------- sampling


def emitted(sampler: FrameSampler, times: list[float]) -> list[float]:
    return [t for t in times if sampler.should_emit(t)]


@pytest.mark.parametrize("source_fps", [25.0, 29.97, 30.0, 60.0])
def test_sampler_downsamples_to_target_rate(source_fps: float) -> None:
    times = [i / source_fps for i in range(int(source_fps * 10))]  # 10 s of video
    kept = emitted(FrameSampler(2.0), times)
    assert len(kept) == 20
    gaps = np.diff(kept)
    assert gaps.min() >= 0.5 - 1 / source_fps and gaps.max() <= 0.5 + 1 / source_fps


def test_sampler_hits_exact_slots_despite_float_rounding() -> None:
    kept = emitted(FrameSampler(2.0), [i / 30 for i in range(31)])
    assert kept == pytest.approx([0.0, 0.5, 1.0])


def test_sampler_keeps_every_frame_when_source_is_slower_than_target() -> None:
    times = [i / 5 for i in range(25)]  # 5 fps source, 10 fps target
    assert emitted(FrameSampler(10.0), times) == times


def test_sampler_does_not_burst_after_a_gap() -> None:
    sampler = FrameSampler(2.0)
    before = emitted(sampler, [i / 10 for i in range(10)])  # 0.0 .. 0.9
    after = emitted(sampler, [30 + i / 10 for i in range(10)])  # stream stalled for ~29 s
    assert len(before) == 2
    assert after == pytest.approx([30.0, 30.5])  # resumes on a fresh grid, no catch-up burst


def test_sampler_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        FrameSampler(0)


# ----------------------------------------------------------------------------- imaging


@pytest.mark.parametrize(
    ("shape", "max_width", "expected"),
    [
        ((1080, 1920, 3), 1280, (720, 1280, 3)),
        ((2160, 3840, 3), 1280, (720, 1280, 3)),
        ((1000, 1500, 3), 1280, (853, 1280, 3)),  # rounds the height
        ((480, 854, 3), 1280, (480, 854, 3)),  # never upscales
        ((720, 1280, 3), 1280, (720, 1280, 3)),  # exactly at the limit
        ((1080, 1080, 3), 640, (640, 640, 3)),
    ],
)
def test_resize_keeps_aspect_ratio(shape, max_width, expected) -> None:
    frame = np.zeros(shape, dtype=np.uint8)
    out = resize_to_max_width(frame, max_width)
    assert out.shape == expected
    if shape[1] <= max_width:
        assert out is frame  # untouched, no copy


def test_jpeg_quality_affects_size_and_decodes() -> None:
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    low, high = encode_jpeg(frame, 30), encode_jpeg(frame, 95)
    assert low[:2] == high[:2] == b"\xff\xd8"  # JPEG magic
    assert len(low) < len(high)
    decoded = cv2.imdecode(np.frombuffer(high, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == frame.shape


# ----------------------------------------------------------------------------- sources


@pytest.mark.parametrize(
    ("url", "kind", "location"),
    [
        ("rtsp://cam.local:554/stream1", SourceKind.RTSP, "rtsp://cam.local:554/stream1"),
        ("https://example.org/traffic.mp4", SourceKind.HTTP, "https://example.org/traffic.mp4"),
        ("file:///data/videos/a%20b.mp4", SourceKind.FILE, "/data/videos/a b.mp4"),
        ("/data/videos/x.webm", SourceKind.FILE, "/data/videos/x.webm"),
        ("x.webm", SourceKind.FILE, str(Path("/data/videos") / "x.webm")),
    ],
)
def test_resolve_source(url: str, kind: SourceKind, location: str) -> None:
    source = resolve_source(url, Path("/data/videos"))
    assert source.kind is kind
    assert source.location == location
    assert source.loops == (kind is SourceKind.FILE)


@pytest.mark.parametrize("url", ["", "   ", "ftp://example.org/x.mp4"])
def test_resolve_source_rejects_bad_urls(url: str) -> None:
    with pytest.raises(ValueError):
        resolve_source(url, Path("/data/videos"))


def test_config_validation() -> None:
    assert IngestConfig().sample_fps == 2.0
    with pytest.raises(ValueError):
        IngestConfig(jpeg_quality=0)
    with pytest.raises(ValueError):
        IngestConfig(sample_fps=0)


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAME_SAMPLE_FPS", "5")
    monkeypatch.setenv("FRAME_MAX_WIDTH", "640")
    monkeypatch.setenv("FRAME_RETENTION_MINUTES", "2.5")
    cfg = IngestConfig.from_env()
    assert (cfg.sample_fps, cfg.max_width, cfg.retention_minutes) == (5.0, 640, 2.5)
    assert cfg.metrics_interval_seconds == 30.0


# ----------------------------------------------------------------------------- storage


def test_frame_filename_round_trip() -> None:
    t = datetime(2026, 10, 14, 8, 23, 59, 123456, tzinfo=UTC)
    name = frame_filename(t)
    assert name == "20261014T082359.123456Z.jpg"
    assert ":" not in name
    assert parse_frame_timestamp(name) == t
    assert parse_frame_timestamp("notes.txt") is None


def test_frame_store_writes_atomically(tmp_path: Path) -> None:
    t = datetime(2026, 10, 14, 8, 0, tzinfo=UTC)
    path = FrameStore(tmp_path).write("cam-a", t, b"\xff\xd8jpeg")
    assert path == tmp_path / "cam-a" / "20261014T080000.000000Z.jpg"
    assert path.read_bytes() == b"\xff\xd8jpeg"
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no temp file left


def test_retention_deletes_only_expired_frames(tmp_path: Path) -> None:
    now = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    store = FrameStore(tmp_path)
    old = [store.write("cam-a", now - timedelta(minutes=m), b"x") for m in (11, 30, 600)]
    fresh = [store.write("cam-a", now - timedelta(minutes=m), b"x") for m in (0, 5, 9.9)]
    fresh.append(store.write("cam-b", now - timedelta(seconds=1), b"x"))

    deleted = cleanup_expired_frames(tmp_path, timedelta(minutes=10), now=now)

    assert deleted == 3
    assert not any(p.exists() for p in old)
    assert all(p.exists() for p in fresh)
    assert cleanup_expired_frames(tmp_path, timedelta(minutes=10), now=now) == 0


def test_retention_uses_mtime_for_unrecognised_files(tmp_path: Path) -> None:
    camera_dir = tmp_path / "cam-a"
    camera_dir.mkdir()
    stale_tmp = camera_dir / "20261014T080000.000000Z.jpg.tmp"  # left over from a crash
    stale_tmp.write_bytes(b"x")
    recent_other = camera_dir / "notes.txt"
    recent_other.write_bytes(b"x")
    an_hour_ago = time.time() - 3600
    os.utime(stale_tmp, (an_hour_ago, an_hour_ago))

    assert cleanup_expired_frames(tmp_path, timedelta(minutes=10)) == 1
    assert not stale_tmp.exists() and recent_other.exists()


def test_retention_handles_missing_root(tmp_path: Path) -> None:
    assert cleanup_expired_frames(tmp_path / "nope", timedelta(minutes=10)) == 0


# ----------------------------------------------------------------------------- metrics


def test_metrics_fps_is_per_report_interval() -> None:
    m = CameraMetrics("cam-a")
    m.report(now=100.0)
    for _ in range(60):
        m.frame_read()
    for _ in range(20):
        m.frame_published()
    m.error(RuntimeError("boom"))
    snap = m.report(now=110.0)
    assert (snap.frames_read, snap.frames_published, snap.errors) == (60, 20, 1)
    assert snap.publish_fps == pytest.approx(2.0)
    assert "last_error='boom'" in str(snap)
    assert m.report(now=120.0).publish_fps == 0.0  # nothing new since the last report
