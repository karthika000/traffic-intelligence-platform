"""Service configuration from environment variables (all optional; see .env.example)."""

import os
from dataclasses import dataclass, fields

_ENV = {
    "score_interval_seconds": "ANOMALY_SCORE_INTERVAL_SECONDS",
    "history_windows": "ANOMALY_HISTORY_WINDOWS",
    "max_metric_age_seconds": "ANOMALY_MAX_METRIC_AGE_SECONDS",
    "cooldown_seconds": "ANOMALY_COOLDOWN_SECONDS",
    "clear_ratio": "ANOMALY_CLEAR_RATIO",
    "clear_after": "ANOMALY_CLEAR_AFTER",
    "min_consecutive": "ANOMALY_MIN_CONSECUTIVE",
    "hold_speed_ratio": "ANOMALY_HOLD_SPEED_RATIO",
    "hold_stopped_margin": "ANOMALY_HOLD_STOPPED_MARGIN",
    "max_hold_seconds": "ANOMALY_MAX_HOLD_SECONDS",
    "severity_bounds": "ANOMALY_SEVERITY_BOUNDS",
    "update_min_interval_seconds": "ANOMALY_UPDATE_MIN_SECONDS",
    "system_min_cameras": "ANOMALY_SYSTEM_MIN_CAMERAS",
    "system_frames_ratio": "ANOMALY_SYSTEM_FRAMES_RATIO",
    "system_confidence_ratio": "ANOMALY_SYSTEM_CONFIDENCE_RATIO",
    "system_lookback_seconds": "ANOMALY_SYSTEM_LOOKBACK_SECONDS",
    "warmup_seconds": "ANOMALY_WARMUP_SECONDS",
    "calibration_min_sequences": "ANOMALY_CALIBRATION_MIN_SEQUENCES",
    "recovery_ratio": "ANOMALY_RECOVERY_RATIO",
    "recovery_seconds": "ANOMALY_RECOVERY_SECONDS",
    "camera_refresh_seconds": "CAMERA_REFRESH_SECONDS",
    "model_refresh_seconds": "MODEL_REFRESH_SECONDS",
    "stats_interval_seconds": "ANOMALY_STATS_INTERVAL_SECONDS",
    "congestion_model": "CONGESTION_MODEL_NAME",
    "anomaly_model": "ANOMALY_MODEL_NAME",
    "model_alias": "MODEL_ALIAS",
}


@dataclass(frozen=True)
class AnomalyConfig:
    score_interval_seconds: float = 5.0
    history_windows: int = (
        144  # 12 min of 5 s windows: sequence + the anomaly features' 10 min baseline
    )
    max_metric_age_seconds: float = (
        60.0  # older traffic.metrics (e.g. replayed history) aren't scored
    )
    cooldown_seconds: float = 120.0
    clear_ratio: float = 0.8
    clear_after: int = 3
    min_consecutive: int = 2
    # Hold rule: an active alert stays open while loop or camera speed (mean of the last 3
    # windows) is below hold_speed_ratio x the baseline frozen at its start, or stopped
    # vehicles exceed the frozen baseline by more than hold_stopped_margin; at most
    # max_hold_seconds, so a broken sensor can't keep an alert open forever.
    hold_speed_ratio: float = 0.6
    hold_stopped_margin: float = 1.0
    max_hold_seconds: float = 1800.0
    # Warm-up: no traffic alerts for a camera until it has warmup_seconds of metrics (after its
    # first window, a gap that long, or a restart), so the 10-minute baselines are full. At its
    # end the camera is calibrated from that history if the model lacks a calibration for it
    # (anomaly_service.calibration); with fewer than calibration_min_sequences scorable
    # sequences the warm-up goes on.
    warmup_seconds: float = 600.0
    calibration_min_sequences: int = 60
    # Recovery: an alert with traffic hints ends once loop and camera speed are back to at
    # least recovery_ratio x the frozen baseline (and stopped vehicles / pedestrians within
    # hold_stopped_margin of it) for recovery_seconds, even if the score hasn't cleared. From
    # the first recovered check its updates carry details.recovered_since, so fusion can
    # resolve on a clearing report without waiting for the end.
    recovery_ratio: float = 0.85
    recovery_seconds: float = 60.0
    camera_refresh_seconds: float = 60.0  # re-read per-camera settings (zone_config.anomaly)
    # Severity from score / threshold: below the 1st bound low, then medium, high, critical.
    severity_bounds: str = "2,4,8"
    # An update is published when severity rises or a new hint appears, at most this often.
    update_min_interval_seconds: float = 15.0
    # System-wide degradation: when at least system_min_cameras cameras show a frame drop
    # (frames below system_frames_ratio x their baseline) or a detection-confidence drop
    # (below system_confidence_ratio x baseline) in the same check, one scope="system"
    # camera_degraded anomaly is raised instead of per-camera alerts. New traffic alerts are
    # suppressed while it lasts; traffic alerts that started within system_lookback_seconds
    # before it (or during it) are closed when it clears (after clear_after normal checks).
    system_min_cameras: int = 3
    system_frames_ratio: float = 0.85
    system_confidence_ratio: float = 0.9
    system_lookback_seconds: float = 60.0
    model_refresh_seconds: float = 300.0
    stats_interval_seconds: float = 60.0
    congestion_model: str = "congestion-classifier"
    anomaly_model: str = "traffic-anomaly-detector"
    model_alias: str = "champion"

    @property
    def severity_limits(self) -> tuple[float, float, float]:
        low, medium, high = (float(x) for x in self.severity_bounds.split(","))
        return low, medium, high

    @classmethod
    def from_env(cls) -> "AnomalyConfig":
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            raw = os.getenv(_ENV.get(f.name, ""), "").strip()
            if raw:
                kwargs[f.name] = f.type(raw) if f.type in (int, float, str) else raw
        return cls(**kwargs)
