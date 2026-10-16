"""The output every incident text classifier returns (pyfunc, baseline and rules alike)."""

import json

import numpy as np
import pandas as pd

from ml.nlp.labels import TASKS, LabelMap

OUTPUT_COLUMNS = [
    "incident_type",
    "incident_confidence",
    "severity",
    "severity_confidence",
    "probabilities",  # JSON: {"incident_type": {label: p}, "severity": {label: p}}
]


def texts_from(model_input) -> list[str]:
    """A DataFrame with a "text" column, a Series, or a list of strings -> list of strings."""
    if isinstance(model_input, pd.DataFrame):
        column = "text" if "text" in model_input.columns else model_input.columns[0]
        return model_input[column].astype(str).tolist()
    if isinstance(model_input, pd.Series):
        return model_input.astype(str).tolist()
    if isinstance(model_input, str):
        return [model_input]
    return [str(t) for t in model_input]


def prediction_frame(probs: dict[str, np.ndarray], maps: dict[str, LabelMap]) -> pd.DataFrame:
    """Per-task probability matrices [n, classes] -> one row per text (OUTPUT_COLUMNS)."""
    n = len(next(iter(probs.values())))
    out: dict[str, list] = {c: [] for c in OUTPUT_COLUMNS}
    best = {task: probs[task].argmax(axis=1) for task in TASKS}
    for i in range(n):
        out["incident_type"].append(maps["incident_type"].labels[best["incident_type"][i]])
        out["incident_confidence"].append(float(probs["incident_type"][i].max()))
        out["severity"].append(maps["severity"].labels[best["severity"][i]])
        out["severity_confidence"].append(float(probs["severity"][i].max()))
        out["probabilities"].append(
            json.dumps(
                {
                    task: {
                        label: round(float(p), 4)
                        for label, p in zip(maps[task].labels, probs[task][i], strict=True)
                    }
                    for task in TASKS
                }
            )
        )
    return pd.DataFrame(out, columns=OUTPUT_COLUMNS)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
