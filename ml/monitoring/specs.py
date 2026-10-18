"""What is monitored for each model: features, their kind, and valid ranges for data quality."""

from ml.features import METRIC_COLUMNS, SENSOR_COLUMNS
from ml.monitoring.drift import CheckSpec, Feature

EMBEDDING_DIMS = 16
EMBEDDING_COLUMNS = tuple(f"emb_{i}" for i in range(EMBEDDING_DIMS))

# Valid ranges of the traffic inputs (TrafficMetrics windows joined with loop readings).
INPUT_RANGES = {
    "vehicle_count": (0, 500),
    "avg_speed_kmh": (0, 200),
    "p85_speed_kmh": (0, 250),
    "density": (0, 200),
    "stopped_vehicles": (0, 200),
    "pedestrians_in_roadway": (0, 200),
    "mean_confidence": (0, 1),
    "frames": (0, 200),
    "loop_vehicle_count": (0, 1000),
    "loop_speed_kmh": (0, 200),
    "loop_occupancy": (0, 1),
}
# Default thresholds below were calibrated on quiet windows of the running platform (scores of
# unshifted production data against the reference), with room above the noise. Low-variance
# inputs (frames per window, loop occupancy) need no special threshold: the drift check also
# requires a minimum practical change in their mean (MonitorConfig.min_relative_change).
TRAFFIC_INPUTS = tuple(
    Feature(c, "num", "data", "traffic input", INPUT_RANGES[c], quality=True)
    for c in (*METRIC_COLUMNS, *SENSOR_COLUMNS)
)
# Classifier confidences pile up just below 1: compare their histograms.
CONFIDENCE = {"method": "jensenshannon", "threshold": 0.1}
# Live reports already score ~0.2 against reports_train.csv (held-out phrasings get lower
# confidence); unfamiliar text in a drift test scored ~0.6.
TEXT_CONFIDENCE = {"method": "jensenshannon", "threshold": 0.3}

SPECS: dict[str, CheckSpec] = {
    "vehicle-detector": CheckSpec(
        "vehicle-detector",
        (
            Feature(
                "frame_brightness",
                "num",
                "data",
                "mean grey level of sampled frames",
                (0, 255),
                quality=True,
            ),
            Feature("frame_contrast", "num", "data", "grey-level standard deviation", (0, 255)),
            Feature("confidence", "num", "prediction", "per detected object", (0, 1)),
            Feature("object_class", "cat", "prediction", "class mix of detected objects"),
            Feature(
                "objects_per_frame", "num", "prediction", "objects detected per frame", (0, 300)
            ),
            Feature("no_detections", "cat", "prediction", "frame with no detections"),
        ),
    ),  # fmt: skip
    "congestion-classifier": CheckSpec(
        "congestion-classifier",
        (
            *TRAFFIC_INPUTS,
            Feature("predicted_level", "cat", "prediction", "predicted congestion level"),
            Feature(
                "prediction_confidence",
                "num",
                "prediction",
                "classifier confidence",
                (0, 1),
                **CONFIDENCE,
            ),  # fmt: skip
        ),
        seasonal=True,
    ),
    "traffic-anomaly-detector": CheckSpec(
        "traffic-anomaly-detector",
        (
            *TRAFFIC_INPUTS,
            Feature(
                "anomaly_score_ratio", "num", "prediction", "score / camera threshold", (0, 1e6)
            ),
            Feature(
                "above_threshold",
                "cat",
                "prediction",
                "check at or above the camera's threshold (the alert rate)",
            ),
        ),
        seasonal=True,
    ),  # fmt: skip
    "incident-text-classifier": CheckSpec(
        "incident-text-classifier",
        (
            # Eight classes from a few hundred reports: sampling noise alone reaches ~0.1.
            Feature(
                "predicted_incident_type", "cat", "prediction", "predicted class", threshold=0.15
            ),
            Feature(
                "predicted_severity", "cat", "prediction", "predicted severity", threshold=0.15
            ),
            Feature(
                "incident_confidence",
                "num",
                "prediction",
                "class confidence",
                (0, 1),
                **TEXT_CONFIDENCE,
            ),
            Feature(
                "severity_confidence",
                "num",
                "prediction",
                "severity confidence",
                (0, 1),
                **TEXT_CONFIDENCE,
            ),
            # Live reports use phrasings the training split holds out (by design), so lengths
            # and the unlocated share already differ a little from reports_train.csv.
            Feature(
                "text_length",
                "num",
                "data",
                "characters per report",
                (1, 10_000),
                quality=True,
                threshold=0.3,
            ),
            Feature("unlocated", "cat", "data", "no camera matched", threshold=0.15),
            Feature(
                "report_embedding",
                "embedding",
                "data",
                "report text embedding (classifier encoder, PCA)",
                columns=EMBEDDING_COLUMNS,
            ),
        ),
    ),  # fmt: skip
}
