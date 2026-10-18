"""Production data for the drift checks, and the transforms shared with the reference builders
(so both sides of a comparison are computed the same way).

- vehicle-detector: per-frame and per-object rows from the ``detections`` topic (read back by
  time), plus brightness and contrast of frame files on the shared frames volume.
- congestion-classifier / traffic-anomaly-detector: ``traffic_metrics`` windows joined with the
  latest loop reading (the models' raw inputs) and the service's outputs in
  ``congestion_history`` (level, confidence, anomaly score and threshold).
- incident-text-classifier: ``reports`` with their latest ``nlp_results``.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from ml.features import METRIC_COLUMNS, SENSOR_COLUMNS, attach_sensor_readings

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- vehicle detector


def read_detections(
    bootstrap: str, start: datetime, end: datetime, max_messages: int = 200_000
) -> list[dict]:
    """Detection messages with timestamps in [start, end), read back from the topic by time
    (a throwaway consumer that never commits)."""
    import uuid

    from confluent_kafka import Consumer, KafkaError, TopicPartition

    from traffic_common.topics import Topic

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"ml-monitor-read-{uuid.uuid4().hex[:8]}",
            "enable.auto.commit": False,
            "auto.offset.reset": "latest",
            "enable.partition.eof": True,
        }
    )
    try:
        meta = consumer.list_topics(Topic.DETECTIONS, timeout=10)
        partitions = list(meta.topics[Topic.DETECTIONS].partitions)
        start_ms = int(start.timestamp() * 1000)
        offsets = consumer.offsets_for_times(
            [TopicPartition(Topic.DETECTIONS, p, start_ms) for p in partitions], timeout=10
        )
        live = [tp for tp in offsets if tp.offset >= 0]  # -1: nothing that recent
        if not live:
            return []
        consumer.assign(live)
        done: set[int] = set()
        out: list[dict] = []
        end_ms = int(end.timestamp() * 1000)
        idle = 0
        while len(done) < len(live) and len(out) < max_messages and idle < 10:
            msgs = consumer.consume(num_messages=1000, timeout=1.0)
            if not msgs:
                idle += 1
                continue
            idle = 0
            for m in msgs:
                if m.error():
                    if m.error().code() == KafkaError._PARTITION_EOF:
                        done.add(m.partition())
                    continue
                if m.timestamp()[1] >= end_ms:
                    done.add(m.partition())
                    continue
                try:
                    out.append(json.loads(m.value()))
                except ValueError:
                    continue
        return out
    finally:
        consumer.close()


def detection_rows(detections: list[dict], model_version: str | None = None) -> pd.DataFrame:
    """One row per frame (objects_per_frame, no_detections) and one per detected object
    (confidence, object_class), in one frame; columns not set on a row are missing."""
    frames, objects = [], []
    for d in detections:
        if model_version is not None and str(d.get("model_version")) != str(model_version):
            continue
        ts = pd.Timestamp(d["timestamp"])
        objs = d.get("objects") or []
        frames.append(
            {
                "camera_id": d["camera_id"],
                "ts": ts,
                "objects_per_frame": float(len(objs)),
                "no_detections": "yes" if not objs else "no",
            }
        )
        for o in objs:
            objects.append(
                {
                    "camera_id": d["camera_id"],
                    "ts": ts,
                    "confidence": float(o["confidence"]),
                    "object_class": str(o["label"]),
                }
            )
    return pd.concat([pd.DataFrame(frames), pd.DataFrame(objects)], ignore_index=True)


def frame_stats(
    frames_dir: Path,
    since: datetime,
    max_files: int = 400,
    seed: int = 0,
    until: datetime | None = None,
) -> pd.DataFrame:
    """Brightness (mean grey level) and contrast (grey-level std) of a sample of the frame
    files written in [since, until) (video_ingest keeps ~10 minutes of them)."""
    from PIL import Image

    if not frames_dir.exists():
        return pd.DataFrame(columns=["camera_id", "ts", "frame_brightness", "frame_contrast"])
    cutoff, end = since.timestamp(), until.timestamp() if until else float("inf")
    files = [
        (p, p.stat().st_mtime)
        for cam in frames_dir.iterdir()
        if cam.is_dir()
        for p in cam.glob("*.jpg")
    ]
    files = [(p, t) for p, t in files if cutoff <= t < end]
    random.Random(seed).shuffle(files)
    rows = []
    for path, mtime in files[:max_files]:
        try:
            with Image.open(path) as img:
                img.draft("L", (160, 120))  # decode at reduced size: fast
                grey = np.asarray(img.convert("L"), dtype=np.float32)
        except (OSError, ValueError):
            continue  # removed by retention meanwhile, or half-written
        rows.append(
            {
                "camera_id": path.parent.name,
                "ts": pd.Timestamp(mtime, unit="s", tz="UTC"),
                "frame_brightness": float(grey.mean()),
                "frame_contrast": float(grey.std()),
            }
        )
    return pd.DataFrame(rows, columns=["camera_id", "ts", "frame_brightness", "frame_contrast"])


def last_seen(frame: pd.DataFrame, cameras: list[str]) -> dict[str, datetime | None]:
    if frame.empty or "ts" not in frame:
        return dict.fromkeys(cameras)
    newest = pd.to_datetime(frame["ts"], utc=True).groupby(frame["camera_id"]).max()
    return {c: (newest[c].to_pydatetime() if c in newest.index else None) for c in cameras}


def camera_ids(engine: Engine) -> list[str]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text("SELECT id FROM cameras ORDER BY id"))]


# ----------------------------------------------------------------------------- traffic models

_METRICS_SQL = f"""
SELECT camera_id, window_end AS ts, {", ".join(METRIC_COLUMNS)}
FROM traffic_metrics WHERE window_end >= :since AND window_end < :until
"""
_READINGS_SQL = """
SELECT camera_id, reading_at, vehicle_count, avg_speed_kmh, occupancy
FROM sensor_readings WHERE reading_at >= :since AND reading_at < :until
"""
_OUTPUTS_SQL = """
SELECT camera_id, window_end AS ts, level AS predicted_level, confidence AS prediction_confidence,
       anomaly_score, anomaly_threshold
FROM congestion_history WHERE window_end >= :since AND window_end < :until
"""
RAW_INPUTS = [*METRIC_COLUMNS, *SENSOR_COLUMNS]


def anomaly_outputs(scores: pd.Series, thresholds: pd.Series) -> pd.DataFrame:
    ratio = pd.to_numeric(scores, errors="coerce") / pd.to_numeric(thresholds, errors="coerce")
    above = pd.Series(np.where(ratio.isna(), None, np.where(ratio >= 1.0, "yes", "no")))
    return pd.DataFrame(
        {"anomaly_score_ratio": ratio.to_numpy(), "above_threshold": above.to_numpy()}
    )


def traffic_rows(engine: Engine, start: datetime, end: datetime) -> pd.DataFrame:
    """The traffic models' raw inputs per camera window, with the service's outputs."""
    params = {"since": start, "until": end}
    with engine.connect() as conn:
        metrics = pd.read_sql(text(_METRICS_SQL), conn, params=params)
        readings = pd.read_sql(
            text(_READINGS_SQL), conn, params={"since": start - timedelta(minutes=1), "until": end}
        )
        outputs = pd.read_sql(text(_OUTPUTS_SQL), conn, params=params)
    if metrics.empty:
        return pd.DataFrame(columns=["camera_id", "ts", *RAW_INPUTS])
    raw = attach_sensor_readings(metrics, readings)
    if not outputs.empty:
        outputs["ts"] = pd.to_datetime(outputs["ts"], utc=True)
        outputs = pd.concat(
            [
                outputs.reset_index(drop=True),
                anomaly_outputs(outputs["anomaly_score"], outputs["anomaly_threshold"]),
            ],
            axis=1,
        ).drop(columns=["anomaly_score", "anomaly_threshold"])
        raw = raw.merge(outputs.drop_duplicates(["camera_id", "ts"]), on=["camera_id", "ts"],
                        how="left")  # fmt: skip
    return raw


# ----------------------------------------------------------------------------- text classifier

_REPORTS_SQL = """
SELECT r.id, r.text, r.reported_at AS ts, n.incident_type AS predicted_incident_type,
       n.severity AS predicted_severity, n.incident_confidence, n.severity_confidence,
       n.camera_id AS matched_camera, n.model_version
FROM reports r
JOIN LATERAL (
    SELECT * FROM nlp_results WHERE report_id = r.id ORDER BY processed_at DESC LIMIT 1
) n ON true
WHERE r.reported_at >= :since AND r.reported_at < :until
ORDER BY r.reported_at DESC
"""


def text_features(texts: pd.Series, matched: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_length": texts.astype(str).str.len().astype(float).to_numpy(),
            "unlocated": np.where(matched.isna().to_numpy(), "yes", "no"),
        }
    )


def text_rows(engine: Engine, start: datetime, end: datetime) -> pd.DataFrame:
    """Reports in the window with their latest NLP result (newest first)."""
    with engine.connect() as conn:
        df = pd.read_sql(text(_REPORTS_SQL), conn, params={"since": start, "until": end})
    if df.empty:
        return df
    feats = text_features(df["text"], df["matched_camera"])
    return pd.concat([df.reset_index(drop=True), feats], axis=1)
