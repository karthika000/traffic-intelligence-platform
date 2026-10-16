"""The registered "incident-summarizer": an MLflow pyfunc holding a configuration (generator,
model name, prompt template version, generation settings), not weights. The model itself
loads from the Hugging Face cache; the prompt comes from the ml package logged with it.

Input: a DataFrame with a "context" column (IncidentContext.to_dict() JSON).
Output: headline, summary, generation, latency_ms, faithfulness (JSON)."""

import json
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from ml.summarize.context import IncidentContext
from ml.summarize.summarizer import IncidentSummarizer, SummarizerConfig


class IncidentSummarizerModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.load_config(json.loads(Path(context.artifacts["config"]).read_text()))

    def load_config(self, config: dict) -> "IncidentSummarizerModel":
        self.config = SummarizerConfig(**config)
        self.summarizer = IncidentSummarizer(self.config)
        return self

    def predict(self, context, model_input, params=None):
        rows = []
        for raw in model_input["context"]:
            ctx = IncidentContext.from_dict(json.loads(raw) if isinstance(raw, str) else raw)
            r = self.summarizer.summarize(ctx)
            rows.append(
                {
                    "headline": r.headline,
                    "summary": r.summary,
                    "generation": r.generation,
                    "latency_ms": r.latency_ms,
                    "faithfulness": json.dumps(r.faithfulness),
                }
            )
        return pd.DataFrame(rows)


def write_config(config: SummarizerConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2))
    return path
