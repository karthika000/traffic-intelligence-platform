"""Keyword fallback for nlp_service while no classifier is registered in MLflow. Crude by
design: it keeps the pipeline running and is clearly labelled (model "incident-text-rules")."""

import re

import numpy as np
import pandas as pd

from ml.nlp.labels import label_maps, preprocess
from ml.nlp.predict import prediction_frame, texts_from

_TYPE_KEYWORDS = {
    "accident": r"crash|collision|collided|accident|rtc|overturned|pile.?up|knocked|hit",
    "breakdown": r"broken.?down|breakdown|stalled|broke down|recovery|flat tyre|out of fuel",
    "road_works": r"road ?works|roadworks|resurfacing|workmen|contraflow|temporary lights|works",
    "congestion": r"congestion|queue|queuing|tailback|gridlock|slow|standstill|jam|heavy traffic",
    "weather_hazard": r"flood|ice|icy|snow|fog|fallen tree|storm|spray|water|rain",
    "debris": r"debris|tyre|load|shed|object|spilled|spill|ladder|mattress|rubbish",
    "police_activity": r"police|checkpoint|cordon|officers|incident response|protest|march",
}
_SEVERITY_KEYWORDS = {
    "critical": r"serious|major|all lanes|closed|trapped|injur|fatal|air ambulance|evacuat",
    "high": r"lanes? (?:shut|blocked|closed)|long delays|blocked|severe|heavy",
    "medium": r"delays|lane|slow|caution|minutes",
}


class KeywordIncidentClassifier:
    name = "incident-text-rules"
    version = "rules-1"

    def __init__(self) -> None:
        self.maps = label_maps()
        self._types = {k: re.compile(rf"\b(?:{v})", re.I) for k, v in _TYPE_KEYWORDS.items()}
        self._severity = {k: re.compile(rf"\b(?:{v})", re.I) for k, v in _SEVERITY_KEYWORDS.items()}

    def predict(self, model_input) -> pd.DataFrame:
        texts = [preprocess(t) for t in texts_from(model_input)]
        types, sev = self.maps["incident_type"], self.maps["severity"]
        p_type = np.full((len(texts), len(types)), 0.02)
        p_sev = np.full((len(texts), len(sev)), 0.05)
        for i, text in enumerate(texts):
            hits = {k: len(rx.findall(text)) for k, rx in self._types.items()}
            label = max(hits, key=hits.get) if any(hits.values()) else "none"
            p_type[i, types.to_id[label]] = 0.6
            level = next((k for k, rx in self._severity.items() if rx.search(text)), "low")
            p_sev[i, sev.to_id[level]] = 0.6
        p_type /= p_type.sum(axis=1, keepdims=True)
        p_sev /= p_sev.sum(axis=1, keepdims=True)
        return prediction_frame({"incident_type": p_type, "severity": p_sev}, self.maps)
