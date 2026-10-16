"""The fused confidence and its breakdown (stored with every incident so the dashboard can
explain it). All parts are in [0, 1]:

vision = base + (1 - base) x (w_ratio x ratio_score + w_duration x duration_score)
         ratio_score    = (peak score/threshold - 1) / (vision_strong_ratio - 1), clipped
         duration_score = anomaly duration / vision_full_duration_seconds, clipped
         x degraded_vision_factor when the anomalies were seen during a system degradation
text   = w_count x (1 - exp(-recent supporting reports / count_scale))
         + w_sources x min(distinct recent sources, full) / full    (recent: text_burst_seconds)
         + w_classifier x mean classifier confidence + w_match x mean camera-match confidence
agreement = how well the anomaly hints support the reported type (0 unless both are present)

fused  = w_primary x max(vision, text) + w_secondary x min(vision, text) + w_agreement x agreement
"""

import math
from collections import Counter
from datetime import datetime, timedelta

from fusion_service.config import FusionConfig
from fusion_service.model import AnomalyEvidence, ReportEvidence
from fusion_service.types import hint_agreement, type_from_hints, vote_type


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def vision_component(anomalies: list[AnomalyEvidence], now: datetime, cfg: FusionConfig) -> dict:
    traffic = [a for a in anomalies if a.traffic]
    if not traffic:
        return {"score": 0.0, "present": False}
    clean = [a for a in traffic if not a.degraded]
    used, degraded = (clean, False) if clean else (traffic, True)
    peak = max(a.peak_ratio for a in used)
    seconds = max(((a.ended_at or now) - a.started_at).total_seconds() for a in used)
    ratio_score = _clip((peak - 1.0) / max(cfg.vision_strong_ratio - 1.0, 1e-6))
    duration_score = _clip(seconds / cfg.vision_full_duration_seconds)
    raw = cfg.vision_base + (1 - cfg.vision_base) * (
        cfg.vision_ratio_weight * ratio_score + cfg.vision_duration_weight * duration_score
    )
    hints = sorted({h for a in used for h in a.hints})
    vision_type = type_from_hints(hints)
    return {
        "score": round(raw * (cfg.degraded_vision_factor if degraded else 1.0), 4),
        "present": True,
        "raw_score": round(raw, 4),
        "degraded": degraded,
        "peak_ratio": round(peak, 3),
        "ratio_score": round(ratio_score, 4),
        "active_seconds": round(seconds, 1),
        "duration_score": round(duration_score, 4),
        "anomalies": len(used),
        "active": any(a.active for a in used),
        "hints": hints,
        "type": vision_type.value if vision_type else None,
    }


def text_component(reports: list[ReportEvidence], now: datetime, cfg: FusionConfig) -> dict:
    """Only reports that support the leading type count (congestion reports at half weight,
    being a consequence); other attached reports are listed but don't add confidence, so
    unrelated background reports can't build up a false incident. The count and sources use
    only the last ``text_burst_seconds``: what matters is how many people are reporting it
    now, not how many reports a quiet camera collects over the whole window."""
    active = [r for r in reports if r.stage == "active"]
    cleared = len(reports) - len(active)
    if not active:
        return {"score": 0.0, "present": False, "cleared_reports": cleared}
    votes: dict[str, float] = {}
    for r in active:
        votes[r.incident_type] = votes.get(r.incident_type, 0.0) + r.confidence
    text_type = vote_type(votes)
    leading = text_type.value if text_type else active[0].incident_type

    def support(r: ReportEvidence) -> float:
        # Congestion reports half-support a specific incident (its consequence); a congestion
        # incident is supported by congestion reports only.
        if r.incident_type == leading:
            return 1.0
        return 0.5 if r.incident_type == "congestion" else 0.0

    weighted = [(r, support(r)) for r in active if support(r) > 0]
    since = now - timedelta(seconds=cfg.text_burst_seconds)
    recent = [(r, w) for r, w in weighted if r.ts >= since]
    n = sum(w for _, w in recent)
    total = sum(w for _, w in weighted) or 1.0
    sources = Counter(r.source or "unknown" for r, _ in recent)
    classifier = sum(r.confidence * w for r, w in weighted) / total
    match = sum(r.camera_confidence * w for r, w in weighted) / total
    count_score = 1 - math.exp(-n / cfg.text_count_scale)
    source_score = min(len(sources), cfg.text_full_sources) / cfg.text_full_sources
    score = (
        cfg.text_weight_count * count_score
        + cfg.text_weight_sources * source_score
        + cfg.text_weight_classifier * classifier
        + cfg.text_weight_match * match
    )
    return {
        "score": round(_clip(score), 4),
        "present": True,
        "type": leading,
        "supporting_reports": round(sum(w for _, w in weighted), 1),
        "recent_supporting_reports": round(n, 1),
        "attached_reports": len(active),
        "sources": dict(sources),
        "n_sources": len(sources),
        "count_score": round(count_score, 4),
        "source_score": round(source_score, 4),
        "classifier": round(classifier, 4),
        "camera_match": round(match, 4),
        "unlocated_reports": sum(r.unlocated for r, _ in weighted),
        "cleared_reports": cleared,
        "type_votes": {t: round(v, 3) for t, v in sorted(votes.items())},
    }


def agreement_component(vision: dict, text: dict) -> dict:
    both = vision.get("present") and text.get("present")
    score = hint_agreement(vision.get("hints", []), text.get("type")) if both else 0.0
    return {
        "score": round(score, 4),
        "both_modalities": bool(both),
        "vision_type": vision.get("type"),
        "text_type": text.get("type"),
    }


def fuse(vision: dict, text: dict, agreement: dict, cfg: FusionConfig) -> dict:
    v, t, a = vision["score"], text["score"], agreement["score"]
    primary, secondary = max(v, t), min(v, t)
    contributions = {
        "primary": round(cfg.weight_primary * primary, 4),
        "secondary": round(cfg.weight_secondary * secondary, 4),
        "agreement": round(cfg.weight_agreement * a, 4),
    }
    return {
        "fused": round(_clip(sum(contributions.values())), 4),
        "primary_modality": "vision" if v >= t else "text",
        "weights": {
            "primary": cfg.weight_primary,
            "secondary": cfg.weight_secondary,
            "agreement": cfg.weight_agreement,
        },
        "contributions": contributions,
    }


def breakdown(
    anomalies: list[AnomalyEvidence],
    reports: list[ReportEvidence],
    now: datetime,
    cfg: FusionConfig,
) -> dict:
    vision = vision_component(anomalies, now, cfg)
    text = text_component(reports, now, cfg)
    agreement = agreement_component(vision, text)
    return {
        "vision": vision,
        "text": text,
        "agreement": agreement,
        "fusion": fuse(vision, text, agreement, cfg),
    }
