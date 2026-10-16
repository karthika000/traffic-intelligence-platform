"""The common incident type space (traffic_common.enums.IncidentType) and how each modality maps
into it; how well two pieces of evidence agree; severities.

- NLP incident types are already IncidentType values.
- Anomaly hints map to a type: stopped vehicles plus a speed drop -> accident, stopped vehicles
  alone -> breakdown, a speed drop alone -> congestion, pedestrians in the road -> police
  activity. camera_degraded / sensor_dropout are equipment faults, not incidents.
- When both are present the NLP type wins: text says *what* happened, the time series only
  that traffic changed.
"""

from collections.abc import Iterable

from traffic_common.enums import AnomalyHint, IncidentType, Severity

T = IncidentType
TRAFFIC_HINTS = {
    AnomalyHint.SPEED_DROP.value,
    AnomalyHint.STOPPED_VEHICLES.value,
    AnomalyHint.PEDESTRIAN_IN_ROAD.value,
}
EQUIPMENT_HINTS = {AnomalyHint.CAMERA_DEGRADED.value, AnomalyHint.SENSOR_DROPOUT.value}
SEVERITIES = [s.value for s in Severity]


def is_traffic(hints: Iterable[str]) -> bool:
    """Whether an anomaly's hints describe traffic (an incident) rather than equipment."""
    return bool(TRAFFIC_HINTS & set(hints))


def type_from_hints(hints: Iterable[str]) -> IncidentType | None:
    h = set(hints)
    stopped, slow = AnomalyHint.STOPPED_VEHICLES.value in h, AnomalyHint.SPEED_DROP.value in h
    if stopped and slow:
        return T.ACCIDENT
    if stopped:
        return T.BREAKDOWN
    if slow:
        return T.CONGESTION
    if AnomalyHint.PEDESTRIAN_IN_ROAD.value in h:
        return T.POLICE_ACTIVITY
    return None


# How strongly an anomaly hint supports a reported incident type (0-1). A speed drop is
# consistent with anything that blocks the road; stopped vehicles point at a stoppage.
HINT_SUPPORT: dict[str, dict[IncidentType, float]] = {
    AnomalyHint.SPEED_DROP.value: {
        T.CONGESTION: 1.0,
        T.ACCIDENT: 0.8,
        T.BREAKDOWN: 0.8,
        T.ROAD_WORKS: 0.8,
        T.WEATHER_HAZARD: 0.8,
        T.DEBRIS: 0.7,
        T.POLICE_ACTIVITY: 0.6,
    },
    AnomalyHint.STOPPED_VEHICLES.value: {
        T.ACCIDENT: 1.0,
        T.BREAKDOWN: 1.0,
        T.DEBRIS: 0.6,
        T.CONGESTION: 0.6,
        T.ROAD_WORKS: 0.5,
        T.WEATHER_HAZARD: 0.5,
        T.POLICE_ACTIVITY: 0.5,
    },
    AnomalyHint.PEDESTRIAN_IN_ROAD.value: {
        T.POLICE_ACTIVITY: 0.9,
        T.ACCIDENT: 0.8,
        T.DEBRIS: 0.4,
        T.ROAD_WORKS: 0.4,
    },
}


def hint_agreement(hints: Iterable[str], incident_type: IncidentType | str | None) -> float:
    """Agreement between the vision evidence and the text's incident type (0-1)."""
    if incident_type is None:
        return 0.0
    t = T(incident_type)
    return max(
        (HINT_SUPPORT.get(h, {}).get(t, 0.2) for h in hints if h in TRAFFIC_HINTS), default=0.0
    )


# Compatibility of two reported types: can a report of type b belong to an incident of type a?
# Congestion is the consequence of almost anything, so it fits every incident.
_RELATED = {
    frozenset({T.ACCIDENT, T.BREAKDOWN}): 0.6,
    frozenset({T.ACCIDENT, T.DEBRIS}): 0.5,
    frozenset({T.ACCIDENT, T.POLICE_ACTIVITY}): 0.5,
    frozenset({T.BREAKDOWN, T.DEBRIS}): 0.4,
    frozenset({T.WEATHER_HAZARD, T.ACCIDENT}): 0.4,
    frozenset({T.WEATHER_HAZARD, T.DEBRIS}): 0.4,
    frozenset({T.ROAD_WORKS, T.DEBRIS}): 0.3,
}


def type_compatibility(a: IncidentType | str, b: IncidentType | str) -> float:
    a, b = T(a), T(b)
    if a == b:
        return 1.0
    if T.NONE in (a, b):
        return 0.0
    if T.CONGESTION in (a, b):
        return 0.7
    return _RELATED.get(frozenset({a, b}), 0.1)


def vote_type(votes: dict[str, float], congestion_weight: float = 0.5) -> IncidentType | None:
    """The leading reported type: summed classifier confidence per type. Congestion counts
    less, because it is usually the consequence of the real incident."""
    if not votes:
        return None
    weighted = {t: v * (congestion_weight if t == T.CONGESTION else 1.0) for t, v in votes.items()}
    return T(max(weighted, key=weighted.get))


def severity_rank(severity: str | None) -> int:
    return SEVERITIES.index(severity) if severity in SEVERITIES else -1


def max_severity(*severities: str | None) -> str:
    ranked = [s for s in severities if s in SEVERITIES]
    return max(ranked, key=SEVERITIES.index) if ranked else Severity.LOW.value


CONGESTION_SEVERITY = {
    "free_flow": "low",
    "moderate": "low",
    "heavy": "medium",
    "standstill": "high",
}


def vision_severity(anomaly_severity: str | None, congestion: str | None) -> str | None:
    """Severity from the time series: the anomaly's (score/threshold) severity, capped at high
    unless traffic is at a standstill (a big score alone can come from a glitchy channel), and
    at least what the congestion level implies."""
    if anomaly_severity is None and congestion is None:
        return None
    sev = anomaly_severity
    if sev == Severity.CRITICAL.value and congestion != "standstill":
        sev = Severity.HIGH.value
    return max_severity(sev, CONGESTION_SEVERITY.get(congestion or "", None))


def robust_text_severity(severities: list[str], min_reports: int) -> str | None:
    """The highest severity at least ``min_reports`` reports agree on (at or above it), so one
    misclassified report can't set the incident's severity; the max if fewer reports."""
    if not severities:
        return None
    ranked = sorted((severity_rank(s) for s in severities if s in SEVERITIES), reverse=True)
    if not ranked:
        return None
    k = min(min_reports, len(ranked))
    return SEVERITIES[ranked[k - 1]]
