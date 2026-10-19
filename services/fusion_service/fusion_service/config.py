"""Service configuration from environment variables (all optional; see .env.example).
Every field can be set as FUSION_<FIELD NAME IN UPPER CASE>, e.g. FUSION_WINDOW_SECONDS."""

import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class FusionConfig:
    # --- correlation
    window_seconds: float = 600.0  # evidence on a camera this close in time is one incident
    max_message_age_seconds: float = 900.0  # older messages (replays) don't create incidents
    # --- when text "fires" on its own: a burst of same-type reports from several sources
    text_burst_seconds: float = 180.0
    min_text_reports: int = 4
    min_text_sources: int = 2
    min_report_confidence: float = 0.5  # classifier confidence below this is ignored
    min_compatibility: float = 0.5  # a report joins an incident if its type fits this well
    # --- start time: the first evidence supporting the incident's type, in the run of such
    # evidence (gaps of at most this long) that led to the incident firing
    start_gap_seconds: float = 120.0
    # --- unlocated reports (no camera match): attach when a place entity matches one of an
    # open incident's place entities at least this well (fuzzy ratio 0-1)
    unlocated_match_score: float = 0.88
    unlocated_camera_confidence_factor: float = 0.8
    # --- fused confidence: strongest modality, the other one, and their agreement
    weight_primary: float = 0.6
    weight_secondary: float = 0.25
    weight_agreement: float = 0.15
    # vision: anomaly score / threshold (1 -> 0, vision_strong_ratio -> 1) and duration
    vision_strong_ratio: float = 4.0
    vision_full_duration_seconds: float = 120.0
    vision_base: float = 0.3  # any traffic anomaly is worth this much
    vision_ratio_weight: float = 0.6
    vision_duration_weight: float = 0.4
    degraded_vision_factor: float = 0.3  # vision during a system-wide degradation
    # text: number of reports, distinct sources, classifier and camera-match confidence
    text_count_scale: float = 3.0  # count score = 1 - exp(-n / scale)
    text_weight_count: float = 0.4
    text_weight_sources: float = 0.25
    text_weight_classifier: float = 0.2
    text_weight_match: float = 0.15
    text_full_sources: int = 3
    # --- lifecycle
    confirm_min_agreement: float = 0.5  # both modalities agree at least this well -> confirmed
    vision_confirm_min: float = 0.4  # ...if each modality is at least this strong (a single
    text_confirm_min: float = 0.65  # background report on a camera with an anomaly isn't)
    strong_single: float = 0.9  # or one modality alone scores this much
    quiet_seconds: float = 300.0  # no new evidence for this long -> resolved
    clear_min_seconds: float = 20.0  # with a clearing report: wait this long after the anomaly
    near_baseline_ratio: float = 0.8  # recent camera speed >= ratio x baseline = back to normal
    metrics_recent_seconds: float = 60.0
    # A camera's anomalies can create an incident on their own only once it has had this long
    # of uninterrupted metrics (gaps under metrics_gap_seconds), the same warm-up anomaly_service
    # applies before alerting, so a camera that just came online can't raise vision-only
    # incidents. Restarts reload the recent metrics from Postgres.
    vision_warmup_seconds: float = 600.0
    metrics_gap_seconds: float = 60.0
    min_severity_reports: int = 2
    publish_confidence_delta: float = 0.1  # republish when confidence moves this much
    # --- loop
    tick_seconds: float = 5.0
    stats_interval_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "FusionConfig":
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            raw = os.getenv(f"FUSION_{f.name.upper()}", "").strip()
            if raw:
                kwargs[f.name] = f.type(raw) if f.type in (int, float, str) else raw
        return cls(**kwargs)
