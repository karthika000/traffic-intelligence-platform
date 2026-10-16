"""Rule-based fallbacks used by anomaly_service when no champion model is registered.

Same interface as the MLflow pyfunc models: raw rows in (features.RAW_COLUMNS, the
camera's recent history), one result row per camera out.
"""

import json
import warnings

import numpy as np
import pandas as pd

from ml.features import DEFAULT_REFERENCE, base_features
from ml.hints import derive_hints
from ml.labels import weak_congestion_labels

RULES_VERSION = "rules-1"


class RuleBasedCongestion:
    name = "congestion-rules"
    version = RULES_VERSION

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        base = base_features(raw, {})
        last = base.groupby("camera_id").tail(1)
        levels = weak_congestion_labels(last)
        return pd.DataFrame(
            {
                "camera_id": last["camera_id"].to_numpy(),
                "ts": last["ts"].to_numpy(),
                "level": levels.to_numpy(),
                "confidence": np.nan,
            }
        )


class RuleBasedAnomaly:
    """Score 1.0 = threshold. Components (each scaled so 1.0 means alarming):

    - speed drop: latest loop/camera speed vs the camera's recent median, >= 50% drop
    - stopped vehicles: >= 3 inside the roadway
    - pedestrians in road: >= 3 and well above the camera's recent normal
    - sensor dropout: no loop reading in the last windows after having one before
    - camera degraded: mean confidence < 0.2 or under a third of the usual frames
    """

    name = "anomaly-rules"
    version = RULES_VERSION
    threshold = 1.0

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        with warnings.catch_warnings():  # medians of all-missing columns (no loop readings yet)
            warnings.simplefilter("ignore", RuntimeWarning)
            rows = [self._score_camera(df) for _, df in raw.groupby("camera_id")]
        return pd.DataFrame(rows)

    def _score_camera(self, df: pd.DataFrame) -> dict:
        df = df.sort_values("ts")
        history, recent = df.iloc[:-3], df.iloc[-3:]
        last = df.iloc[-1]
        components, observed, expected = {}, {}, {}

        for col in ("loop_speed_kmh", "avg_speed_kmh"):
            base = history[col].median() if len(history) >= 6 else np.nan
            now = recent[col].mean()
            if pd.notna(base) and pd.notna(now) and base > 5:
                components[col] = max(0.0, (base - now) / base) / 0.5
                observed[col], expected[col] = float(now), float(base)
        stopped = float(recent["stopped_vehicles"].fillna(0).max())
        components["stopped_vehicles"] = stopped / 3
        observed["stopped_vehicles"], expected["stopped_vehicles"] = stopped, 0.0
        peds = float(recent["pedestrians_in_roadway"].fillna(0).min())
        usual = float(history["pedestrians_in_roadway"].fillna(0).median()) if len(history) else 0.0
        components["pedestrians_in_roadway"] = peds / max(3.0, 2 * usual + 2)
        observed["pedestrians_in_roadway"], expected["pedestrians_in_roadway"] = peds, usual
        had_sensor = history["loop_speed_kmh"].notna().any() if len(history) else False
        sensor_missing = bool(had_sensor and recent["loop_speed_kmh"].isna().all())
        components["sensor_present"] = 1.0 if sensor_missing else 0.0
        conf = recent["mean_confidence"].mean()
        frames = recent["frames"].mean()
        usual_frames = (
            history["frames"].median() if len(history) else DEFAULT_REFERENCE.medians["frames"]
        )
        degraded = (pd.notna(conf) and conf < 0.2) or (
            pd.notna(frames) and pd.notna(usual_frames) and frames < usual_frames / 3
        )
        components["mean_confidence"] = 1.0 if degraded else 0.0
        if pd.notna(conf):
            observed["mean_confidence"] = float(conf)
            expected["mean_confidence"] = (
                float(history["mean_confidence"].median()) if len(history) else 0.6
            )

        score = max(components.values())
        total = sum(components.values()) or 1.0
        contributions = {k: round(v / total, 4) for k, v in components.items() if v > 0}
        top = max(components, key=components.get)
        return {
            "camera_id": last["camera_id"],
            "ts": last["ts"],
            "valid": len(df) >= 4,
            "score": float(score),
            "threshold": self.threshold,
            "is_anomaly": bool(score >= self.threshold and len(df) >= 4),
            "contributions": json.dumps(contributions),
            "top_feature": top,
            "observed": float(observed.get(top, np.nan)) if top in observed else np.nan,
            "expected": float(expected.get(top, np.nan)) if top in expected else np.nan,
            "hints": json.dumps(derive_hints(contributions, observed, expected, sensor_missing)),
        }
