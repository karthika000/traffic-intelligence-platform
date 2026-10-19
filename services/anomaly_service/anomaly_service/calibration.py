"""Per-camera calibration at the end of a camera's warm-up, and per-camera signal exclusions.

The anomaly model scales each feature's reconstruction error by that camera's normal error and
compares the result with that camera's threshold, both fitted on validation data at training
time. A camera the model never saw (a fresh install trains on synthetic data only, or a camera
added later) falls back to the global scales and threshold, which come from other cameras'
noise. A camera whose own footage is noisier, say a panning view or a traffic light, then
scores far above that threshold most of the time.

``calibrate_camera`` fills that gap the way training does, from the camera's own warm-up
history: error and level scales at the model's quantiles for the features computed from the
video, and a threshold at the model's percentile of the resulting scores. Two limits:

- Only video features. What a view produces (panning, a traffic light, parked vans in the
  roadway polygon) is specific to the camera; the loop detectors are the same sensor at every
  site, and ten minutes of their 15 s readings overstate their noise. Calibrating them too
  hid three of the four e2e scenarios, which reach the platform through the loops.
- Never below the model's global value. Ten minutes is a small sample: a camera may become
  less sensitive than the global calibration (its footage is noisier), never more.

It only fills what the model lacks, so a camera calibrated at training time keeps its own.

``zone_config["anomaly"]`` adjusts one camera (see ``CameraSettings``): ``ignore`` switches off
signal groups whose readings that camera can't be trusted for (their scales become infinite,
so they never score), and ``threshold_factor`` scales its threshold.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.features import AE_FEATURES, GLOBAL, LEVEL_FEATURES
from ml.models.anomaly import score_frame, score_parts, window_levels

log = logging.getLogger(__name__)

# Signal groups a camera can ignore -> the model features (AE inputs and level terms) they feed.
SIGNAL_FEATURES: dict[str, tuple[str, ...]] = {
    "camera_speed": ("avg_speed_kmh_rel", "p85_speed_kmh_rel", "camera_speed_present"),
    "stopped_vehicles": ("stopped_vehicles", "stopped_excess"),
    "pedestrians": ("pedestrians_in_roadway", "pedestrians_excess"),
    "density": ("density_rel",),
    "vehicle_count": ("vehicle_count_rel",),
}
# Features computed from the loop detectors: they keep the model's scales.
SENSOR_FEATURES = {
    "loop_speed_kmh_rel",
    "loop_vehicle_count_rel",
    "loop_occupancy_rel",
    "sensor_present",
}
LEVEL_FLOORS = [0.1] * (len(LEVEL_FEATURES) - 2) + [1.0, 1.0]  # as ml.models.anomaly.calibrate


@dataclass(frozen=True)
class CameraSettings:
    """Per-camera anomaly settings from ``cameras.zone_config["anomaly"]``."""

    ignore: tuple[str, ...] = ()
    threshold_factor: float = 1.0

    @classmethod
    def from_zone_config(cls, zone_config: dict | None) -> "CameraSettings":
        raw = (zone_config or {}).get("anomaly") or {}
        ignore = tuple(s for s in raw.get("ignore", []) if s in SIGNAL_FEATURES)
        unknown = set(raw.get("ignore", [])) - set(ignore)
        if unknown:
            log.warning("zone_config anomaly.ignore: unknown signals %s ignored", sorted(unknown))
        factor = float(raw.get("threshold_factor", 1.0))
        return cls(ignore, factor if factor > 0 else 1.0)

    def ignores(self, signal: str) -> bool:
        return signal in self.ignore

    @property
    def ignored_features(self) -> set[str]:
        return {f for s in self.ignore for f in SIGNAL_FEATURES[s]}


@dataclass
class CalibrationResult:
    camera_id: str
    sequences: int
    own_scales: bool  # the model already had scales for this camera
    own_threshold: bool
    threshold: float
    ignored: tuple[str, ...]
    fitted_threshold: float | None = None  # the warm-up's own percentile, before the floor

    def describe(self) -> str:
        if self.own_threshold:
            source = "the model's"
        else:
            fitted = self.fitted_threshold or 0.0
            floor = ", raised to the global floor" if self.threshold > fitted else ""
            source = f"warm-up: {fitted:.2f} over {self.sequences} sequences{floor}"
        return (
            f"threshold {self.threshold:.2f} ({source}), scales from "
            f"{'the model' if self.own_scales else 'the warm-up'}, "
            f"ignored={list(self.ignored) or '-'}"
        )


def python_model(predictor):
    """The AnomalyPyfunc behind an MLflow pyfunc, or None (e.g. the rule-based fallback)."""
    unwrap = getattr(predictor, "unwrap_python_model", None)
    if unwrap is None:
        return None
    try:
        py = unwrap()
    except Exception:
        return None
    return py if hasattr(py, "calibration") and hasattr(py, "normalizer") else None


def calibrate_camera(
    py,
    pristine,
    camera_id: str,
    raw: pd.DataFrame,
    settings: CameraSettings,
    *,
    min_sequences: int = 60,
) -> CalibrationResult | None:
    """Fit what the model lacks for ``camera_id`` from its raw history and apply ``settings``.

    ``pristine`` is the model's calibration as loaded (a copy); the result is written into
    ``py.calibration``, the loaded model's own. Returns None when the history has fewer than
    ``min_sequences`` scorable sequences and something had to be fitted."""
    cal, cfg = py.calibration, py.cfg
    own_scales = camera_id in pristine.error_scale
    own_threshold = camera_id in pristine.thresholds
    sequences = 0
    if own_scales:
        error_scale = list(pristine.error_scale[camera_id])
        level_scale = list(
            pristine.level_scale.get(camera_id) or pristine.level_scale.get(GLOBAL) or []
        )
    if not (own_scales and own_threshold):
        sf = score_frame(raw, py.refs, py.normalizer, py.model, cfg, calibration=None)
        ends = np.where(~np.isnan(sf.scores))[0]
        sequences = len(ends)
        if sequences < min_sequences:
            return None
        errors = sf.errors[ends]  # unscaled without a calibration
        levels = window_levels(sf.base, ends, cfg.score_steps)
    if not own_scales:
        error_scale = _video_scales(
            np.percentile(errors, cfg.scale_quantile, axis=0),
            pristine.error_scale[GLOBAL],
            AE_FEATURES,
        )
        level_scale = (
            _video_scales(
                np.maximum(np.percentile(levels, cfg.level_quantile, axis=0), LEVEL_FLOORS),
                pristine.level_scale[GLOBAL],
                LEVEL_FEATURES,
            )
            if pristine.level_scale
            else []
        )
    ignored = settings.ignored_features
    for i, feature in enumerate(AE_FEATURES):
        if feature in ignored:
            error_scale[i] = math.inf  # scaled error 0: the feature never scores
    for i, feature in enumerate(LEVEL_FEATURES):
        if feature in ignored and level_scale:
            level_scale[i] = math.inf
    cal.error_scale[camera_id] = error_scale
    if level_scale:
        cal.level_scale[camera_id] = level_scale
    fitted = None
    if own_threshold:
        threshold = pristine.thresholds[camera_id]
    else:
        scores = score_parts(errors, levels, np.full(sequences, camera_id), cal, cfg).score
        fitted = float(np.percentile(scores, cfg.threshold_percentile))
        # At least the global threshold, for the same reason as the scales.
        threshold = max(fitted, pristine.thresholds[GLOBAL])
    threshold *= settings.threshold_factor
    cal.thresholds[camera_id] = threshold
    return CalibrationResult(
        camera_id, sequences, own_scales, own_threshold, threshold, settings.ignore, fitted
    )


def _video_scales(own, global_scale, features) -> list[float]:
    """The camera's own scale for video features, at least the global one; the global scale
    for loop-detector features."""
    return [
        float(g) if f in SENSOR_FEATURES else float(max(o, g))
        for f, o, g in zip(features, own, global_scale, strict=True)
    ]
