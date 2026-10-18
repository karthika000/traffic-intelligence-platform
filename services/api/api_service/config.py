"""API configuration from environment variables (all optional; see .env.example)."""

import os
from dataclasses import dataclass, fields
from pathlib import Path

_ENV = {
    "cors_origins": "API_CORS_ORIGINS",
    "annotated_dir": "ANNOTATED_DIR",
    "scenarios_dir": "SCENARIOS_DIR",
    "mjpeg_fps": "API_MJPEG_FPS",
    "mjpeg_max_fps": "API_MJPEG_MAX_FPS",
    "ws_metrics_interval_seconds": "API_WS_METRICS_INTERVAL_SECONDS",
    "ws_heartbeat_seconds": "API_WS_HEARTBEAT_SECONDS",
    "ws_queue_size": "API_WS_QUEUE_SIZE",
    "ws_max_drops": "API_WS_MAX_DROPS",
    "mlflow_cache_seconds": "API_MLFLOW_CACHE_SECONDS",
    "max_page_size": "API_MAX_PAGE_SIZE",
    "drift_reports_dir": "DRIFT_REPORTS_DIR",
}


@dataclass(frozen=True)
class ApiConfig:
    # Origins allowed to call the API from a browser (the dashboard), comma-separated.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    annotated_dir: Path = Path("/data/annotated")  # cv_inference's <camera>/latest.jpg
    scenarios_dir: Path = Path("/app/scenarios")  # event_simulator's scenario YAML files
    mjpeg_fps: float = 5.0  # default stream rate; ?fps= overrides up to mjpeg_max_fps
    mjpeg_max_fps: float = 15.0
    # Live WebSocket: at most one traffic.metrics message per camera per this many seconds,
    # a heartbeat after this many idle seconds, a per-client queue of this many messages
    # (the oldest are dropped when a client is slow), and a slow client is disconnected
    # after this many drops in a row without catching up.
    ws_metrics_interval_seconds: float = 2.0
    ws_heartbeat_seconds: float = 15.0
    ws_queue_size: int = 500
    ws_max_drops: int = 5000
    mlflow_cache_seconds: float = 60.0
    max_page_size: int = 200
    drift_reports_dir: Path = Path("/data/drift_reports")  # ml-monitor's Evidently HTML reports

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @classmethod
    def from_env(cls) -> "ApiConfig":
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            raw = os.getenv(_ENV.get(f.name, ""), "").strip()
            if not raw:
                continue
            if f.type is Path or f.type == "Path":
                kwargs[f.name] = Path(raw)
            elif f.type in (int, float, str):
                kwargs[f.name] = f.type(raw)
        return cls(**kwargs)
