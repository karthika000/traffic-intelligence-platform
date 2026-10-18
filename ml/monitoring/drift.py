"""Drift and data quality checks with Evidently: production data against a model's reference.

A check compares one model's current window with its reference snapshot, feature by feature:

- **Numbers:** normalised Wasserstein distance (the earth mover's distance in units of the
  reference's standard deviation).
- **Categories:** Jensen-Shannon distance between the two distributions.
- **Embeddings:** the ROC AUC of a classifier trained to tell reference from current vectors
  (0.5 = indistinguishable).

Each feature gets a score and drifts when the score reaches its threshold. The model is
*drifted* when the share of drifted features reaches the share threshold, *warning* when some
feature drifted or a data quality check failed, *ok* otherwise, and *no_data* without enough
current rows. Data quality covers missing values (against the reference's own missing share),
values outside each input's valid range, and stale cameras. Everything is also written as one
Evidently HTML report.
"""

import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ml.monitoring.config import MonitorConfig

log = logging.getLogger(__name__)

Kind = Literal["num", "cat", "embedding"]
Status = Literal["ok", "warning", "drifted", "no_data"]
METHODS = {"num": "wasserstein", "cat": "jensenshannon", "embedding": "model_roc_auc"}


@dataclass(frozen=True)
class Feature:
    name: str
    kind: Kind
    drift_type: Literal["data", "prediction"]
    description: str = ""
    valid_range: tuple[float, float] | None = None  # data quality: values outside are invalid
    quality: bool = False  # an input: checked for missing values
    columns: tuple[str, ...] = ()  # embedding columns
    # Calibrated defaults for this feature (DRIFT_FEATURE_THRESHOLDS still overrides them).
    # method "jensenshannon" on a number compares histograms: for scores piled up near 1, where
    # a standard-deviation-normalised distance blows up on tiny shifts.
    method: str | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class CheckSpec:
    model_name: str
    features: tuple[Feature, ...]
    seasonal: bool = False  # reference rows from the same time of day (column "ts")

    def feature(self, name: str) -> Feature:
        return next(f for f in self.features if f.name == name)


@dataclass
class FeatureResult:
    feature: str
    kind: str
    drift_type: str
    method: str
    score: float | None
    threshold: float
    drifted: bool
    reference_rows: int
    current_rows: int
    reference_summary: dict[str, Any] = field(default_factory=dict)
    current_summary: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


@dataclass
class QualityIssue:
    kind: str  # missing_values | out_of_range | stale_camera
    target: str  # feature or camera
    value: float
    limit: float
    message: str


@dataclass
class CheckResult:
    model_name: str
    model_version: str
    status: Status
    drift_share: float
    share_threshold: float
    window_start: datetime
    window_end: datetime
    features: list[FeatureResult]
    quality: list[QualityIssue]
    reference_rows: int
    current_rows: int
    check_id: uuid.UUID = field(default_factory=uuid.uuid4)
    html_path: Path | None = None
    reference: dict[str, Any] = field(default_factory=dict)  # reference snapshot metadata
    note: str | None = None

    @property
    def drifted_features(self) -> list[FeatureResult]:
        return [f for f in self.features if f.drifted]

    @property
    def drift_type(self) -> str:
        """ "data" if any input drifted, else "prediction" (the alert's drift type)."""
        return (
            "data" if any(f.drift_type == "data" for f in self.drifted_features) else "prediction"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["check_id"] = str(self.check_id)
        d["html_path"] = str(self.html_path) if self.html_path else None
        d["window_start"] = self.window_start.isoformat()
        d["window_end"] = self.window_end.isoformat()
        d["drifted_features"] = [f.feature for f in self.drifted_features]
        return _jsonable(d)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, datetime | pd.Timestamp):
        return value.isoformat()
    return value


# ----------------------------------------------------------------------------- reference slicing


def seasonal_slice(
    reference: pd.DataFrame,
    window_end: datetime,
    hours: float,
    min_rows: int = 500,
    timezone: str = "UTC",
) -> tuple[pd.DataFrame, str | None]:
    """Reference rows from the same kind of day (weekday or weekend, in ``timezone``) whose
    time of day is within +/- ``hours`` of the window's, if there are at least ``min_rows``;
    else the same kind of day at any hour; else the whole reference. A weekday rush hour is
    no reference for a Saturday afternoon (traffic follows a weekly pattern too)."""
    if hours <= 0 or "ts" not in reference.columns or reference.empty:
        return reference, None
    ts = pd.to_datetime(reference["ts"], utc=True).dt.tz_convert(timezone)
    end = pd.Timestamp(window_end).tz_convert(timezone)
    weekend = end.dayofweek >= 5
    day = "weekend" if weekend else "weekday"
    same_day = ((ts.dt.dayofweek >= 5) == weekend).to_numpy()
    minutes = ts.dt.hour * 60 + ts.dt.minute
    diff = (minutes - (end.hour * 60 + end.minute)).abs()
    near = (np.minimum(diff, 1440 - diff) <= hours * 60).to_numpy()
    if (same_day & near).sum() >= min_rows:
        mask = same_day & near
        return reference[
            mask
        ], f"{day}s at the same time of day +/- {hours:g} h ({mask.sum()} rows)"
    if same_day.sum() >= min_rows:
        return reference[same_day], f"{day}s at any time of day ({same_day.sum()} rows)"
    return reference, f"whole reference: it has no {day} data at this time of day"


# ----------------------------------------------------------------------------- summaries


def _summary(values: pd.Series, kind: str) -> dict[str, Any]:
    if kind == "cat":
        shares = values.astype(str).value_counts(normalize=True).head(10)
        return {"shares": {k: round(float(v), 4) for k, v in shares.items()}}
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return {}
    return {
        "mean": round(float(v.mean()), 4),
        "median": round(float(v.median()), 4),
        "p10": round(float(v.quantile(0.1)), 4),
        "p90": round(float(v.quantile(0.9)), 4),
    }


def _js_distance(ref: pd.Series, cur: pd.Series) -> float:
    from scipy.spatial.distance import jensenshannon

    keys = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    p = ref.astype(str).value_counts(normalize=True).reindex(keys, fill_value=0).to_numpy()
    q = cur.astype(str).value_counts(normalize=True).reindex(keys, fill_value=0).to_numpy()
    return float(jensenshannon(p, q, base=2))


# ----------------------------------------------------------------------------- drift


def _prepare(frame: pd.DataFrame, spec: CheckSpec) -> pd.DataFrame:
    """The spec's columns, typed; a column the frame doesn't have is all missing."""
    out = pd.DataFrame(index=frame.index)

    def col(name: str) -> pd.Series:
        return frame[name] if name in frame else pd.Series(None, index=frame.index, dtype=object)

    for f in spec.features:
        if f.kind == "embedding":
            for c in f.columns:
                out[c] = pd.to_numeric(col(c), errors="coerce").astype(float)
        elif f.kind == "cat":
            out[f.name] = col(f.name).map(lambda v: None if pd.isna(v) else str(v)).astype(object)
        else:
            out[f.name] = pd.to_numeric(col(f.name), errors="coerce").astype(float)
    return out.reset_index(drop=True)


def _require_practical_change(
    result: FeatureResult, ref: pd.Series, cur: pd.Series, config: MonitorConfig
) -> None:
    """A low-variance feature (reference std / |mean| below ``low_variance_cv``) also needs its
    mean to move by at least ``min_relative_change`` of the reference mean to count as drifted.

    The Wasserstein distance is measured in reference standard deviations, so when the spread
    is tiny (frames per window, loop occupancy) a shift of a percent or two, or a current window
    that is merely narrower than the reference band, scores far above the threshold."""
    ref, cur = ref.dropna(), cur.dropna()
    mean = float(ref.mean())
    if not mean or config.low_variance_cv <= 0:
        return
    cv = float(ref.std()) / abs(mean)
    if cv >= config.low_variance_cv:
        return
    change = abs(float(cur.mean()) - mean) / abs(mean)
    if change < config.min_relative_change:
        result.drifted = False
        result.note = (
            f"low-variance feature (std {cv:.0%} of the mean): the mean moved {change:.1%}, "
            f"under the {config.min_relative_change:.0%} minimum practical change"
        )


def drift_report(
    spec: CheckSpec,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: MonitorConfig,
    html_path: Path | None = None,
) -> list[FeatureResult]:
    """Per-feature drift scores with Evidently, plus the HTML report when ``html_path``."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import EmbeddingsDrift, MissingValueCount, OutRangeValueCount, ValueDrift
    from evidently.metrics.embeddings import ModelDriftMethod

    ref, cur = _prepare(reference, spec), _prepare(current, spec)
    results: list[FeatureResult] = []
    metrics, index = [], {}
    for f in spec.features:
        threshold = config.threshold(spec.model_name, f.name, f.kind, f.threshold)
        method = f.method or METHODS[f.kind]
        base = {
            "feature": f.name,
            "kind": f.kind,
            "drift_type": f.drift_type,
            "method": method,
            "threshold": threshold,
        }
        if f.kind == "embedding":
            r_rows = int(ref[list(f.columns)].notna().all(axis=1).sum())
            c_rows = int(cur[list(f.columns)].notna().all(axis=1).sum())
        else:
            r_rows, c_rows = int(ref[f.name].notna().sum()), int(cur[f.name].notna().sum())
        result = FeatureResult(
            **base,
            score=None,
            drifted=False,
            reference_rows=r_rows,
            current_rows=c_rows,
        )
        if f.kind != "embedding":
            result.reference_summary = _summary(ref[f.name].dropna(), f.kind)
            result.current_summary = _summary(cur[f.name].dropna(), f.kind)
        results.append(result)
        if r_rows < config.min_rows or c_rows < config.min_rows:
            result.note = f"too few rows (reference {r_rows}, current {c_rows})"
            continue
        if f.kind == "num" and method == "wasserstein" and float(np.nanstd(ref[f.name])) == 0.0:
            # A constant reference has no scale for the Wasserstein distance: compare the
            # value distributions as categories instead.
            result.method = METHODS["cat"]
            result.score = _js_distance(ref[f.name].dropna(), cur[f.name].dropna())
            result.drifted = result.score >= config.threshold(spec.model_name, f.name, "cat")
            result.threshold = config.threshold(spec.model_name, f.name, "cat")
            result.note = "constant in the reference; compared as categories"
            continue
        index[len(metrics)] = result
        if f.kind == "embedding":
            metrics.append(
                EmbeddingsDrift(
                    embeddings_name=f.name,
                    drift_method=ModelDriftMethod(threshold=threshold, bootstrap=False),
                )
            )
        else:
            metrics.append(ValueDrift(column=f.name, method=method, threshold=threshold))

    quality_metrics = []
    for f in spec.features:
        if f.quality and f.kind != "embedding":
            quality_metrics.append(MissingValueCount(column=f.name))
        if f.valid_range is not None and f.kind == "num":
            lo, hi = f.valid_range
            quality_metrics.append(OutRangeValueCount(column=f.name, left=lo, right=hi))

    embeddings = {f.name: list(f.columns) for f in spec.features if f.kind == "embedding"}
    definition = DataDefinition(
        numerical_columns=[
            c for f in spec.features for c in ((f.name,) if f.kind == "num" else f.columns)
        ],
        categorical_columns=[f.name for f in spec.features if f.kind == "cat"],
        embeddings=embeddings or None,
    )
    if not metrics and not quality_metrics:
        return results
    report = Report([*metrics, *quality_metrics])
    snapshot = report.run(
        Dataset.from_pandas(cur, data_definition=definition),
        Dataset.from_pandas(ref, data_definition=definition),
    )
    values = snapshot.dict()["metrics"]
    for i, result in index.items():
        value = values[i].get("value")
        score = float(value) if isinstance(value, int | float) else None
        result.score = score
        result.drifted = score is not None and score >= result.threshold
        if result.drifted and result.method == "wasserstein":
            _require_practical_change(result, ref[result.feature], cur[result.feature], config)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
    return results


# ----------------------------------------------------------------------------- data quality


def quality_issues(
    spec: CheckSpec,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: MonitorConfig,
    last_seen: dict[str, datetime] | None = None,
    now: datetime | None = None,
) -> list[QualityIssue]:
    """Missing values (beyond the reference's own share), out-of-range values, stale cameras.

    ``last_seen``: each expected camera's newest data (None if none in the window)."""
    issues: list[QualityIssue] = []
    for f in spec.features:
        if f.kind == "embedding" or f.name not in current.columns or current.empty:
            continue
        if f.quality:
            ref_missing = float(reference[f.name].isna().mean()) if f.name in reference else 0.0
            cur_missing = float(current[f.name].isna().mean())
            limit = ref_missing + config.missing_tolerance
            if cur_missing > limit:
                issues.append(
                    QualityIssue(
                        "missing_values",
                        f.name,
                        round(cur_missing, 4),
                        round(limit, 4),
                        f"{f.name}: {cur_missing:.0%} missing (reference {ref_missing:.0%})",
                    )
                )
        if f.valid_range is not None:
            values = pd.to_numeric(current[f.name], errors="coerce").dropna()
            if len(values):
                lo, hi = f.valid_range
                share = float(((values < lo) | (values > hi)).mean())
                if share > config.out_of_range_max:
                    issues.append(
                        QualityIssue(
                            "out_of_range",
                            f.name,
                            round(share, 4),
                            config.out_of_range_max,
                            f"{f.name}: {share:.1%} of values outside [{lo:g}, {hi:g}]",
                        )
                    )
    if last_seen is not None and now is not None:
        for camera, seen in sorted(last_seen.items()):
            age = (now - seen).total_seconds() if seen is not None else math.inf
            if age > config.stale_camera_seconds:
                issues.append(
                    QualityIssue(
                        "stale_camera",
                        camera,
                        round(age, 1) if math.isfinite(age) else -1.0,
                        config.stale_camera_seconds,
                        f"{camera}: no data for {age:.0f}s"
                        if math.isfinite(age)
                        else f"{camera}: no data in the window",
                    )
                )
    return issues


# ----------------------------------------------------------------------------- verdict


def verdict(
    features: list[FeatureResult],
    quality: list[QualityIssue],
    current_rows: int,
    config: MonitorConfig,
) -> tuple[Status, float]:
    """Overall status and the share of (evaluated) features that drifted."""
    scored = [f for f in features if f.score is not None]
    if current_rows < config.min_rows or not scored:
        return "no_data", 0.0
    share = sum(f.drifted for f in scored) / len(scored)
    if share >= config.share_threshold:
        return "drifted", share
    if share > 0 or quality:
        return "warning", share
    return "ok", share


def run_check(
    spec: CheckSpec,
    model_version: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: MonitorConfig,
    window: tuple[datetime, datetime],
    *,
    reports_dir: Path | None = None,
    last_seen: dict[str, datetime] | None = None,
    reference_meta: dict[str, Any] | None = None,
    current_rows: int | None = None,
) -> CheckResult:
    """The full check: reference slicing, drift, data quality and the verdict."""
    start, end = window
    note = None
    if spec.seasonal:
        reference, note = seasonal_slice(
            reference, end, config.reference_hours, timezone=config.timezone
        )
    check_id = uuid.uuid4()
    html_path = reports_dir / f"{check_id}.html" if reports_dir is not None else None
    rows = len(current) if current_rows is None else current_rows
    if rows < config.min_rows:
        features = [
            FeatureResult(
                f.name,
                f.kind,
                f.drift_type,
                f.method or METHODS[f.kind],
                None,
                config.threshold(spec.model_name, f.name, f.kind, f.threshold),
                False,
                len(reference),
                rows,
                note="too few current rows",
            )  # fmt: skip
            for f in spec.features
        ]
        html_path = None
    else:
        features = drift_report(spec, reference, current, config, html_path)
    quality = quality_issues(spec, reference, current, config, last_seen, end)
    status, share = verdict(features, quality, rows, config)
    return CheckResult(
        model_name=spec.model_name,
        model_version=model_version,
        status=status,
        drift_share=share,
        share_threshold=config.share_threshold,
        window_start=start,
        window_end=end,
        features=features,
        quality=quality,
        reference_rows=len(reference),
        current_rows=rows,
        check_id=check_id,
        html_path=html_path if html_path is not None and html_path.exists() else None,
        reference=reference_meta or {},
        note=note,
    )
