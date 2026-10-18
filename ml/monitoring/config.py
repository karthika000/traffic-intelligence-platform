"""ml-monitor settings from environment variables (all optional; see .env.example)."""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

MONITORED_MODELS = (
    "vehicle-detector",
    "congestion-classifier",
    "traffic-anomaly-detector",
    "incident-text-classifier",
)

_ENV = {
    "interval_minutes": "DRIFT_CHECK_INTERVAL_MINUTES",
    "window_minutes": "DRIFT_WINDOW_MINUTES",
    "text_window_minutes": "DRIFT_TEXT_WINDOW_MINUTES",
    "models": "DRIFT_MODELS",
    "share_threshold": "DRIFT_SHARE_THRESHOLD",
    "num_threshold": "DRIFT_NUM_THRESHOLD",
    "cat_threshold": "DRIFT_CAT_THRESHOLD",
    "embedding_threshold": "DRIFT_EMBEDDING_THRESHOLD",
    "feature_thresholds": "DRIFT_FEATURE_THRESHOLDS",
    "min_rows": "DRIFT_MIN_ROWS",
    "missing_tolerance": "DRIFT_MISSING_TOLERANCE",
    "out_of_range_max": "DRIFT_OUT_OF_RANGE_MAX",
    "low_variance_cv": "DRIFT_LOW_VARIANCE_CV",
    "min_relative_change": "DRIFT_MIN_RELATIVE_CHANGE",
    "stale_camera_seconds": "DRIFT_STALE_CAMERA_SECONDS",
    "reference_hours": "DRIFT_REFERENCE_HOURS",
    "reports_dir": "DRIFT_REPORTS_DIR",
    "reports_keep": "DRIFT_REPORTS_KEEP",
    "auto_reference": "DRIFT_AUTO_REFERENCE",
    "auto_retrain": "DRIFT_AUTO_RETRAIN",
    "run_at_start": "DRIFT_RUN_AT_START",
    "max_texts": "DRIFT_MAX_TEXTS",
    "frames_dir": "FRAMES_DIR",
    "reference_root": "DRIFT_REFERENCE_DIR",
    "experiment": "DRIFT_MLFLOW_EXPERIMENT",
    "mlflow_html_every_minutes": "DRIFT_MLFLOW_HTML_EVERY_MINUTES",
    "threads": "MONITOR_THREADS",
    "timezone": "DISPLAY_TIMEZONE",
}


def _bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class MonitorConfig:
    interval_minutes: float = 15.0  # how often ml-monitor runs every check
    window_minutes: float = 15.0  # production data compared with the reference: the last N min
    # Reports arrive about 6 a minute: 15 minutes (~90) is too few for a stable class mix.
    text_window_minutes: float = 30.0
    models: tuple[str, ...] = MONITORED_MODELS
    # A feature drifts when its score reaches its threshold: normalised Wasserstein distance
    # for numbers, Jensen-Shannon distance for categories, the ROC AUC of a classifier telling
    # reference from current embeddings. A model drifts when the share of drifted features
    # reaches share_threshold; fewer drifted features (or a data quality issue) is a warning.
    share_threshold: float = 0.5
    num_threshold: float = 0.1
    cat_threshold: float = 0.1
    embedding_threshold: float = 0.55
    # Per-feature overrides: "feature=0.2" or "model:feature=0.2", comma-separated.
    feature_thresholds: str = ""
    min_rows: int = 30  # fewer current rows than this: no verdict ("no_data")
    missing_tolerance: float = 0.1  # missing share may exceed the reference's by this much
    out_of_range_max: float = 0.01  # share of values outside the valid range allowed
    # A number whose reference spread is small (std below low_variance_cv x |mean|) only drifts
    # if its mean also moves by min_relative_change of the reference mean. 0 = off.
    low_variance_cv: float = 0.25
    min_relative_change: float = 0.05
    stale_camera_seconds: float = 120.0  # a camera with no data for this long is stale
    # Compare with reference rows from the same kind of day (weekday/weekend) and time of day
    # (+/- N hours) when the reference has enough of them, so the daily and weekly traffic
    # pattern isn't reported as drift. 0 = off.
    reference_hours: float = 1.0
    timezone: str = "Europe/London"  # local time for "time of day" and weekday/weekend
    reports_dir: Path = Path("/data/drift_reports")
    reports_keep: int = 100  # HTML reports kept per model (all are also in MLflow)
    auto_reference: bool = True  # build a missing reference for a champion when checked
    auto_retrain: bool = False  # on a drift alert: retrain (True) or recommend retraining
    run_at_start: bool = True
    max_texts: int = 600  # current reports embedded per text check
    frames_dir: Path = Path("/data/frames")
    reference_root: Path = Path("data/ml/reference")
    experiment: str = "drift-monitoring"
    # HTML reports go to MLflow for every warning or drifted check, and for ok checks at most
    # once per this many minutes per model (0 = every check). Metrics are always logged.
    mlflow_html_every_minutes: float = 60.0
    threads: int = 2
    overrides: dict[str, float] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.interval_minutes <= 0 or self.window_minutes <= 0:
            raise ValueError("DRIFT_CHECK_INTERVAL_MINUTES and DRIFT_WINDOW_MINUTES must be > 0")
        if not 0 < self.share_threshold <= 1:
            raise ValueError("DRIFT_SHARE_THRESHOLD must be in (0, 1]")
        unknown = set(self.models) - set(MONITORED_MODELS)
        if unknown:
            raise ValueError(f"DRIFT_MODELS: unknown models {sorted(unknown)}")
        parsed = {}
        for item in filter(None, (s.strip() for s in self.feature_thresholds.split(","))):
            key, _, value = item.partition("=")
            if not value:
                raise ValueError(f"DRIFT_FEATURE_THRESHOLDS: expected key=value, got {item!r}")
            parsed[key.strip()] = float(value)
        object.__setattr__(self, "overrides", parsed)

    def threshold(self, model: str, feature: str, kind: str, default: float | None = None) -> float:
        """DRIFT_FEATURE_THRESHOLDS, else the feature's calibrated default, else by kind."""
        for key in (f"{model}:{feature}", feature):
            if key in self.overrides:
                return self.overrides[key]
        if default is not None:
            return default
        return {
            "num": self.num_threshold,
            "cat": self.cat_threshold,
            "embedding": self.embedding_threshold,
        }[kind]

    @classmethod
    def from_env(cls, **overrides) -> "MonitorConfig":
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            raw = os.getenv(_ENV.get(f.name, ""), "").strip()
            if not raw or f.name == "overrides":
                continue
            if f.type in (Path, "Path"):
                kwargs[f.name] = Path(raw)
            elif f.type in (bool, "bool"):
                kwargs[f.name] = _bool(raw)
            elif f.type in (int, "int"):
                kwargs[f.name] = int(raw)
            elif f.type in (float, "float"):
                kwargs[f.name] = float(raw)
            elif f.name == "models":
                kwargs[f.name] = tuple(m.strip() for m in raw.split(",") if m.strip())
            else:
                kwargs[f.name] = raw
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)
