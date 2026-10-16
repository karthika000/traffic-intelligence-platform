"""Service configuration from environment variables (all optional; see .env.example)."""

import os
from dataclasses import dataclass, fields

_ENV = {
    "registered_model": "SUMMARIZER_MODEL_NAME",
    "model_alias": "MODEL_ALIAS",
    "model_refresh_seconds": "MODEL_REFRESH_SECONDS",
    "debounce_seconds": "SUMMARY_DEBOUNCE_SECONDS",
    "timeout_seconds": "SUMMARY_TIMEOUT_SECONDS",
    "default_model": "SUMMARY_DEFAULT_MODEL",
    "default_prompt": "SUMMARY_DEFAULT_PROMPT",
    "catch_up_hours": "SUMMARY_CATCH_UP_HOURS",
    "stats_interval_seconds": "SUMMARY_STATS_INTERVAL_SECONDS",
}


@dataclass(frozen=True)
class SummarizerServiceConfig:
    registered_model: str = "incident-summarizer"
    model_alias: str = "champion"
    model_refresh_seconds: float = 300.0
    # Updates to one incident within this many seconds are summarised once (the latest).
    debounce_seconds: float = 10.0
    # Generation budget per summary; past it the template is used.
    timeout_seconds: float = 20.0
    # Used until a configuration is registered as @champion.
    default_model: str = "google/flan-t5-base"
    default_prompt: str = "v1"
    # On startup, summarise incident versions from this far back that have no summary yet.
    catch_up_hours: float = 1.0
    tick_seconds: float = 1.0
    stats_interval_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "SummarizerServiceConfig":
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            raw = os.getenv(_ENV.get(f.name, ""), "").strip()
            if raw:
                kwargs[f.name] = f.type(raw) if f.type in (int, float, str) else raw
        return cls(**kwargs)
