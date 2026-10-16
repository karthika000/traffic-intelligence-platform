"""The registered "incident-text-classifier": an MLflow pyfunc around either the fine-tuned
DistilBERT (tokenizer, encoder, heads) or the TF-IDF baseline, plus the label maps.

Input: a DataFrame with a "text" column (or a list of strings).
Output: ml.nlp.predict.OUTPUT_COLUMNS, one row per text."""

import json
from pathlib import Path

import joblib
import mlflow.pyfunc

from ml.nlp.baseline import TfidfBaseline
from ml.nlp.labels import LabelMap
from ml.nlp.predict import prediction_frame, texts_from


def save_baseline_dir(model: TfidfBaseline, maps: dict[str, LabelMap], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "baseline.joblib")
    (out_dir / "config.json").write_text(
        json.dumps(
            {"kind": "tfidf", "label_maps": {t: m.to_dict() for t, m in maps.items()}}, indent=2
        )
    )
    return out_dir


class IncidentTextClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.load_dir(Path(context.artifacts["model_dir"]))

    def load_dir(self, model_dir: Path) -> "IncidentTextClassifier":
        config = json.loads((model_dir / "config.json").read_text())
        self.kind = config["kind"]
        self.maps = {t: LabelMap.from_dict(m) for t, m in config["label_maps"].items()}
        if self.kind == "transformer":
            from ml.nlp import transformer

            self.model, self.tokenizer, self.cfg = transformer.load(model_dir)
        else:
            self.model = joblib.load(model_dir / "baseline.joblib")
        return self

    def predict_probabilities(self, texts: list[str]):
        if self.kind == "transformer":
            from ml.nlp import transformer

            return transformer.predict_proba(self.model, self.tokenizer, texts, self.cfg.max_length)
        return self.model.predict_proba(texts)

    def predict(self, context, model_input, params=None):
        return prediction_frame(self.predict_probabilities(texts_from(model_input)), self.maps)
