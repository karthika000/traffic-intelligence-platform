"""Turn an anomaly's contributing features into human-meaningful hints."""

# feature -> (hint, direction that makes it that hint: "low" or "high" vs expected)
_FEATURE_HINTS = {
    "avg_speed_kmh": ("speed_drop", "low"),
    "p85_speed_kmh": ("speed_drop", "low"),
    "loop_speed_kmh": ("speed_drop", "low"),
    "speed_ratio": ("speed_drop", "low"),
    "loop_speed_ratio": ("speed_drop", "low"),
    "stopped_vehicles": ("stopped_vehicles", "high"),
    "density": ("stopped_vehicles", "high"),
    "pedestrians_in_roadway": ("pedestrian_in_road", "high"),
    "sensor_present": ("sensor_dropout", "low"),
    "loop_vehicle_count": ("sensor_dropout", "low"),
    "loop_occupancy": ("sensor_dropout", "low"),
    "mean_confidence": ("camera_degraded", "low"),
    "frames": ("camera_degraded", "low"),
    "camera_speed_present": ("camera_degraded", "low"),
    # Fewer vehicles than usual is quiet traffic, not a failing camera: vehicle_count maps to
    # no hint. A camera that stops seeing vehicles also loses confidence or frames.
    "stopped_excess": ("stopped_vehicles", "high"),
    "pedestrians_excess": ("pedestrian_in_road", "high"),
}
# The autoencoder's relative inputs (<col>_rel, see ml.features) mean the same thing.
_FEATURE_HINTS.update(
    {
        f"{col}_rel": _FEATURE_HINTS[col]
        for col in (
            "avg_speed_kmh",
            "p85_speed_kmh",
            "loop_speed_kmh",
            "density",
            "loop_vehicle_count",
            "loop_occupancy",
            "frames",
        )
    }
)
MIN_SHARE = 0.1  # features contributing less than this share of the score are ignored
# A pipeline feature only means camera_degraded when it is this far below expected (the same
# limits as anomaly_service's system-wide health check). Frames per window wobble by a frame or
# two, and confidence moves with the scene; neither is degradation.
MIN_RELATIVE_DROP = {"frames": 0.15, "frames_rel": 0.15, "mean_confidence": 0.10}


def derive_hints(
    contributions: dict[str, float],
    observed: dict[str, float],
    expected: dict[str, float],
    sensor_missing: bool = False,
) -> list[str]:
    """Hints ordered by how much of the score they explain.

    A feature only supports its hint if it deviates in the hint's direction (a speed
    *above* expected is not a speed drop). A missing loop sensor always means dropout:
    its filled-in values look normal, so the autoencoder alone can't tell why.
    """
    weight: dict[str, float] = {}
    for feature, share in contributions.items():
        if share < MIN_SHARE or feature not in _FEATURE_HINTS:
            continue
        hint, direction = _FEATURE_HINTS[feature]
        obs, exp = observed.get(feature), expected.get(feature)
        if obs is not None and exp is not None:
            if (direction == "low" and obs >= exp) or (direction == "high" and obs <= exp):
                continue
            if feature in MIN_RELATIVE_DROP and obs > exp * (1 - MIN_RELATIVE_DROP[feature]):
                continue
        weight[hint] = weight.get(hint, 0.0) + share
    if sensor_missing:
        weight["sensor_dropout"] = weight.get("sensor_dropout", 0.0) + 1.0
    if "sensor_dropout" in weight and not sensor_missing:
        del weight["sensor_dropout"]  # low loop counts with a working sensor are just quiet traffic
    return sorted(weight, key=weight.get, reverse=True)
