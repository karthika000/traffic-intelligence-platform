"""Service configuration from environment variables (all optional; see .env.example)."""

import os
from dataclasses import dataclass, fields

_ENV = {
    "classifier_model": "NLP_MODEL_NAME",
    "model_alias": "MODEL_ALIAS",
    "model_refresh_seconds": "MODEL_REFRESH_SECONDS",
    "max_batch": "NLP_MAX_BATCH",
    "ner_enabled": "NLP_NER_ENABLED",
    "ner_model": "NER_MODEL",
    "ner_min_score": "NLP_NER_MIN_SCORE",
    "match_min_score": "NLP_MATCH_MIN_SCORE",
    "match_min_confidence": "NLP_MATCH_MIN_CONFIDENCE",
    "camera_refresh_seconds": "NLP_CAMERA_REFRESH_SECONDS",
    "stats_interval_seconds": "NLP_STATS_INTERVAL_SECONDS",
}


@dataclass(frozen=True)
class NlpConfig:
    classifier_model: str = "incident-text-classifier"
    model_alias: str = "champion"
    model_refresh_seconds: float = 300.0
    # Reports already waiting on reports.text are classified together, up to this many.
    max_batch: int = 32
    ner_enabled: bool = True
    ner_model: str = "dslim/bert-base-NER"
    ner_min_score: float = 0.5
    # Camera matching: an alias must score at least min_score (fuzzy ratio, 0-1); the camera
    # is returned when the confidence (score, reduced if a second camera is close) reaches
    # min_confidence, otherwise camera_id is None.
    match_min_score: float = 0.85
    match_min_confidence: float = 0.7
    camera_refresh_seconds: float = 300.0
    stats_interval_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "NlpConfig":
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            raw = os.getenv(_ENV.get(f.name, ""), "").strip()
            if not raw:
                continue
            if f.type is bool:
                kwargs[f.name] = raw.lower() in ("1", "true", "yes", "on")
            else:
                kwargs[f.name] = f.type(raw) if f.type in (int, float, str) else raw
        return cls(**kwargs)
