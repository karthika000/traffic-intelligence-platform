"""Service configuration from environment variables (all optional; see .env.example)."""

import os
from dataclasses import dataclass
from pathlib import Path


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class IngestConfig:
    sample_fps: float = 5.0  # enough overlap between frames for tracking in cv_inference
    max_width: int = 1280
    jpeg_quality: int = 85
    frames_dir: Path = Path("/data/frames")
    videos_dir: Path = Path("/data/videos")
    retention_minutes: float = 10.0
    cleanup_interval_seconds: float = 60.0
    metrics_interval_seconds: float = 30.0
    camera_refresh_seconds: float = 60.0
    retry_initial_seconds: float = 2.0
    retry_max_seconds: float = 60.0
    source_timeout_seconds: float = 10.0
    drift_mode_poll_seconds: float = 5.0  # how often drift mode (demo degradation) is re-read

    def __post_init__(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("FRAME_SAMPLE_FPS must be > 0")
        if self.max_width < 16:
            raise ValueError("FRAME_MAX_WIDTH must be >= 16")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("FRAME_JPEG_QUALITY must be between 1 and 100")
        if self.retention_minutes <= 0:
            raise ValueError("FRAME_RETENTION_MINUTES must be > 0")
        if self.retry_initial_seconds <= 0 or self.retry_max_seconds < self.retry_initial_seconds:
            raise ValueError("need 0 < SOURCE_RETRY_INITIAL_SECONDS <= SOURCE_RETRY_MAX_SECONDS")

    @classmethod
    def from_env(cls) -> "IngestConfig":
        d = cls.__dataclass_fields__
        return cls(
            sample_fps=_float("FRAME_SAMPLE_FPS", d["sample_fps"].default),
            max_width=_int("FRAME_MAX_WIDTH", d["max_width"].default),
            jpeg_quality=_int("FRAME_JPEG_QUALITY", d["jpeg_quality"].default),
            frames_dir=Path(os.getenv("FRAMES_DIR", str(d["frames_dir"].default))),
            videos_dir=Path(os.getenv("VIDEOS_DIR", str(d["videos_dir"].default))),
            retention_minutes=_float("FRAME_RETENTION_MINUTES", d["retention_minutes"].default),
            cleanup_interval_seconds=_float(
                "FRAME_CLEANUP_INTERVAL_SECONDS", d["cleanup_interval_seconds"].default
            ),
            metrics_interval_seconds=_float(
                "METRICS_LOG_INTERVAL_SECONDS", d["metrics_interval_seconds"].default
            ),
            camera_refresh_seconds=_float(
                "CAMERA_REFRESH_SECONDS", d["camera_refresh_seconds"].default
            ),
            retry_initial_seconds=_float(
                "SOURCE_RETRY_INITIAL_SECONDS", d["retry_initial_seconds"].default
            ),
            retry_max_seconds=_float("SOURCE_RETRY_MAX_SECONDS", d["retry_max_seconds"].default),
            source_timeout_seconds=_float(
                "SOURCE_TIMEOUT_SECONDS", d["source_timeout_seconds"].default
            ),
            drift_mode_poll_seconds=_float(
                "DRIFT_MODE_POLL_SECONDS", d["drift_mode_poll_seconds"].default
            ),
        )
