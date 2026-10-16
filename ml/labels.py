"""Congestion levels and weak labelling rules.

Real data has no congestion ground truth, so it is labelled by thresholds on speed
(relative to the camera's free-flow speed), density (relative to the camera's median)
and stopped vehicles. The synthetic generator labels its noise-free state with the same
rules, so both sources share one definition.
"""

import numpy as np
import pandas as pd

CONGESTION_LEVELS = ["free_flow", "moderate", "heavy", "standstill"]

# Speed ratio = speed / free-flow speed.
STANDSTILL_RATIO = 0.15
HEAVY_RATIO = 0.35
MODERATE_RATIO = 0.65
STOPPED_FOR_STANDSTILL = 3
DENSITY_HEAVY = 2.0  # x median density
DENSITY_MODERATE = 1.6


def congestion_level(speed_ratio, density_ratio, stopped) -> np.ndarray:
    """Vectorised rule: works on scalars, numpy arrays or pandas Series."""
    speed_ratio = np.asarray(speed_ratio, dtype=float)
    density_ratio = np.asarray(density_ratio, dtype=float)
    stopped = np.asarray(stopped, dtype=float)
    level = np.zeros(np.broadcast(speed_ratio, density_ratio, stopped).shape, dtype=int)
    level = np.where((speed_ratio < MODERATE_RATIO) | (density_ratio >= DENSITY_MODERATE), 1, level)
    level = np.where(
        (speed_ratio < HEAVY_RATIO) | ((density_ratio >= DENSITY_HEAVY) & (speed_ratio < 0.6)),
        2,
        level,
    )
    level = np.where(
        (speed_ratio < STANDSTILL_RATIO)
        | ((stopped >= STOPPED_FOR_STANDSTILL) & (speed_ratio < 0.4)),
        3,
        level,
    )
    return level


def weak_congestion_labels(base: pd.DataFrame) -> pd.Series:
    """Weak labels for rows produced by features.base_features.

    The loop detector's speed is used when present (it measures every vehicle); otherwise
    the camera's tracked speed.
    """
    ratio = np.where(base["sensor_present"] > 0, base["loop_speed_ratio"], base["speed_ratio"])
    levels = congestion_level(ratio, base["density_ratio"], base["stopped_vehicles"])
    return pd.Series(np.asarray(CONGESTION_LEVELS)[levels], index=base.index, name="congestion")
