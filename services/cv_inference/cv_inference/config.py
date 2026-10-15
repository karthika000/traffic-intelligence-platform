"""Service configuration from environment variables (all optional; see .env.example)."""

import os
from dataclasses import dataclass, fields
from pathlib import Path

# Env var for each field. FRAME_SAMPLE_FPS and CAMERA_REFRESH_SECONDS are shared with video_ingest.
_ENV = {
    "model": "CV_MODEL",
    "imgsz": "CV_IMGSZ",
    "confidence": "CV_CONFIDENCE",
    "backend": "CV_BACKEND",
    "threads": "CV_THREADS",
    "workers": "CV_WORKERS",
    "max_frame_age_seconds": "CV_MAX_FRAME_AGE_SECONDS",
    "metrics_window_seconds": "CV_METRICS_WINDOW_SECONDS",
    "speed_window_seconds": "CV_SPEED_WINDOW_SECONDS",
    "stopped_speed_kmh": "CV_STOPPED_SPEED_KMH",
    "stopped_after_seconds": "CV_STOPPED_AFTER_SECONDS",
    "track_lost_seconds": "CV_TRACK_LOST_SECONDS",
    "expected_fps": "FRAME_SAMPLE_FPS",
    "default_frame_width_metres": "CV_DEFAULT_FRAME_WIDTH_METRES",
    "stats_interval_seconds": "CV_STATS_INTERVAL_SECONDS",
    "camera_refresh_seconds": "CAMERA_REFRESH_SECONDS",
    "annotate": "CV_ANNOTATE",
    "weights_dir": "WEIGHTS_DIR",
    "annotated_dir": "ANNOTATED_DIR",
    "registry_model_name": "CV_REGISTRY_MODEL",
    "registry_alias": "CV_REGISTRY_ALIAS",
}
BACKENDS = ("openvino", "pytorch", "auto")


def default_threads() -> int:
    """Half the visible cores, at most 6. Inside the Docker VM every vCPU looks alike, so
    using all of them puts inference threads on efficiency cores and next to video decoding
    and Kafka; measured on a Core Ultra 5 225H, 4-8 threads were fastest and 14 was 80x slower."""
    return max(1, min(6, (os.cpu_count() or 2) // 2))


@dataclass(frozen=True)
class CvConfig:
    model: str = "yolo26n.pt"  # Ultralytics pretrained weights used to bootstrap the registry
    imgsz: int = 480
    confidence: float = 0.35  # objects below this are dropped unless the tracker keeps them
    tracker_confidence: float = 0.1  # ByteTrack also uses low-score boxes to continue tracks
    backend: str = "openvino"  # openvino | pytorch | auto (whichever benchmarks faster)
    threads: int = 0  # 0 = default_threads()
    workers: int = 1  # inference threads, each with its own model instance
    max_frame_age_seconds: float = 5.0
    metrics_window_seconds: float = 5.0
    speed_window_seconds: float = 2.0
    stopped_speed_kmh: float = 3.0
    stopped_after_seconds: float = 5.0
    track_lost_seconds: float = 2.0
    expected_fps: float = 5.0
    default_frame_width_metres: float = 30.0
    stats_interval_seconds: float = 30.0
    camera_refresh_seconds: float = 60.0
    annotate: bool = True
    weights_dir: Path = Path("/weights")
    annotated_dir: Path = Path("/data/annotated")
    registry_model_name: str = "vehicle-detector"
    registry_alias: str = "champion"

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"CV_BACKEND must be one of {BACKENDS}")
        if not 0 < self.confidence < 1 or self.imgsz < 32 or self.workers < 1:
            raise ValueError("need 0 < CV_CONFIDENCE < 1, CV_IMGSZ >= 32 and CV_WORKERS >= 1")
        if self.threads == 0:
            object.__setattr__(self, "threads", default_threads())

    @property
    def classes(self) -> tuple[str, ...]:
        return ("car", "truck", "bus", "motorcycle", "bicycle", "person")

    @classmethod
    def from_env(cls) -> "CvConfig":
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            raw = os.getenv(_ENV.get(f.name, ""), "").strip()
            if not raw:
                continue
            if f.type is bool:
                kwargs[f.name] = raw.lower() in ("1", "true", "yes", "on")
            elif f.type is int:
                kwargs[f.name] = int(raw)
            elif f.type is float:
                kwargs[f.name] = float(raw)
            elif f.type is Path:
                kwargs[f.name] = Path(raw)
            else:
                kwargs[f.name] = raw
        return cls(**kwargs)
