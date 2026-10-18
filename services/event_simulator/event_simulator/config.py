"""Service configuration from environment variables (all optional; see .env.example)."""

import os
from dataclasses import dataclass, fields
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]  # services/event_simulator (or /app in the image)


@dataclass(frozen=True)
class SimulatorConfig:
    report_interval_seconds: float = 10.0  # mean gap between background reports, all cameras
    sensor_interval_seconds: float = 15.0  # weather + loop detector, per camera
    camera_refresh_seconds: float = 60.0
    scenario_schedule_minutes: float = 0.0  # random scenario every N minutes; 0 = off
    scheduled_time_scale: float = 1.0
    timezone: str = "Europe/London"  # drives the daily traffic pattern
    scenarios_dir: Path = SERVICE_DIR / "scenarios"
    dataset_dir: Path = Path("/data/nlp")
    stats_interval_seconds: float = 60.0
    max_command_age_seconds: float = 600.0  # ignore stale control messages
    drift_mode_poll_seconds: float = 5.0  # how often drift mode (demo report style) is re-read

    def __post_init__(self) -> None:
        if self.report_interval_seconds <= 0 or self.sensor_interval_seconds <= 0:
            raise ValueError("REPORT_INTERVAL_SECONDS and SENSOR_INTERVAL_SECONDS must be > 0")
        if self.scenario_schedule_minutes < 0:
            raise ValueError("SCENARIO_SCHEDULE_MINUTES must be >= 0 (0 disables it)")

    @classmethod
    def from_env(cls) -> "SimulatorConfig":
        env = {
            "report_interval_seconds": "REPORT_INTERVAL_SECONDS",
            "sensor_interval_seconds": "SENSOR_INTERVAL_SECONDS",
            "camera_refresh_seconds": "CAMERA_REFRESH_SECONDS",
            "scenario_schedule_minutes": "SCENARIO_SCHEDULE_MINUTES",
            "scheduled_time_scale": "SCENARIO_TIME_SCALE",
            "timezone": "SIMULATOR_TIMEZONE",
            "scenarios_dir": "SCENARIOS_DIR",
            "dataset_dir": "DATASET_DIR",
            "drift_mode_poll_seconds": "DRIFT_MODE_POLL_SECONDS",
        }
        kwargs = {}
        for f in fields(cls):
            raw = os.getenv(env.get(f.name, ""), "")
            if raw:
                kwargs[f.name] = Path(raw) if f.type in (Path, "Path") else (
                    raw if f.type in (str, "str") else float(raw))  # fmt: skip
        return cls(**kwargs)
