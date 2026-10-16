"""Evaluation for the text models: classification reports and confusion matrices per task,
the worst misclassified examples, location extraction against the labelled mention, and
camera matching accuracy."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ml.nlp.entities import location_spans
from ml.nlp.labels import TASKS, LabelMap
from ml.nlp.matching import CameraMatcher, normalize

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- classification


def classification_metrics(
    probs: dict[str, np.ndarray], y: dict[str, np.ndarray], maps: dict[str, LabelMap], prefix: str
) -> tuple[dict[str, float], dict[str, dict]]:
    """Flat MLflow metrics (accuracy, macro/weighted F1, per-class P/R/F1) and the full
    classification reports per task."""
    metrics, reports = {}, {}
    for task in TASKS:
        pred = probs[task].argmax(axis=1)
        labels = list(range(len(maps[task])))
        report = classification_report(
            y[task],
            pred,
            labels=labels,
            target_names=list(maps[task].labels),
            output_dict=True,
            zero_division=0,
        )
        reports[task] = report
        metrics[f"{prefix}_{task}_accuracy"] = float(accuracy_score(y[task], pred))
        metrics[f"{prefix}_{task}_macro_f1"] = float(f1_score(y[task], pred, average="macro"))
        metrics[f"{prefix}_{task}_weighted_f1"] = float(f1_score(y[task], pred, average="weighted"))
        for label in maps[task].labels:
            for stat, key in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
                metrics[f"{prefix}_{task}_{stat}_{label}"] = float(report[label][key])
    metrics[f"{prefix}_mean_macro_f1"] = float(
        np.mean([metrics[f"{prefix}_{t}_macro_f1"] for t in TASKS])
    )
    return metrics, reports


def confusion_artifacts(
    probs: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    maps: dict[str, LabelMap],
    out: Path,
    title: str,
) -> None:
    """confusion_<task>.csv and .png (rows: true label, columns: predicted)."""
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        labels = list(maps[task].labels)
        cm = confusion_matrix(y[task], probs[task].argmax(axis=1), labels=range(len(labels)))
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / f"confusion_{task}.csv")
        fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(labels), 0.8 + 0.65 * len(labels)))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title}: {task}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )
        fig.tight_layout()
        fig.savefig(out / f"confusion_{task}.png", dpi=110)
        plt.close(fig)


def worst_misclassified(
    df: pd.DataFrame,
    probs: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    maps: dict[str, LabelMap],
    n: int = 50,
) -> pd.DataFrame:
    """Wrong predictions, most confident first (the model was surest and still wrong)."""
    rows = []
    for task in TASKS:
        pred = probs[task].argmax(axis=1)
        for i in np.flatnonzero(pred != y[task]):
            rows.append(
                {
                    "task": task,
                    "text": df["text"].iloc[i],
                    "source": df["source"].iloc[i],
                    "true": maps[task].labels[y[task][i]],
                    "predicted": maps[task].labels[pred[i]],
                    "confidence": round(float(probs[task][i, pred[i]]), 4),
                    "p_true": round(float(probs[task][i, y[task][i]]), 4),
                }
            )
    out = pd.DataFrame(
        rows, columns=["task", "text", "source", "true", "predicted", "confidence", "p_true"]
    )
    return out.sort_values("confidence", ascending=False).head(n).reset_index(drop=True)


def accuracy_by_source(
    df: pd.DataFrame, probs: dict[str, np.ndarray], y: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for source, idx in df.reset_index(drop=True).groupby("source").groups.items():
        idx = np.asarray(idx)
        rows.append(
            {
                "source": source,
                "n": len(idx),
                **{
                    f"{task}_accuracy": float((probs[task][idx].argmax(1) == y[task][idx]).mean())
                    for task in TASKS
                },
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- locations


def _span_of(text: str, mention: str) -> tuple[int, int] | None:
    i = text.find(mention)
    if i < 0:
        i = text.lower().find(mention.lower())
    return (i, i + len(mention)) if i >= 0 else None


def location_extraction_metrics(
    df: pd.DataFrame, ner: list[list[dict]], gazetteer: list[list[dict]]
) -> tuple[dict[str, float], pd.DataFrame]:
    """Extracted LOCATION spans vs the labelled location_mention, for NER alone, the
    gazetteer alone and both combined (what nlp_service uses):
    - exact: a span equals the mention (normalised: case, "&", abbreviations)
    - overlap_recall: a span overlaps the mention
    - token_f1: word overlap between all location spans and the mention
    - precision: share of extracted spans that overlap the mention
    - false_rate: rows without a mention where something was extracted"""
    variants = {
        "ner": [location_spans(t, n, []) if n else [] for t, n in zip(df.text, ner, strict=True)],
        "gazetteer": [location_spans(t, [], g) for t, g in zip(df.text, gazetteer, strict=True)],
        "combined": [
            location_spans(t, n, g) for t, n, g in zip(df.text, ner, gazetteer, strict=True)
        ],
    }
    # NER-only spans: LOC only (ORG/MISC need the gazetteer to confirm them).
    metrics, rows = {}, []
    for name, spans_per_row in variants.items():
        exact = overlap = n_spans = good_spans = with_mention = false_hits = 0
        f1s = []
        for text, mention, spans in zip(df.text, df.location_mention, spans_per_row, strict=True):
            if not isinstance(mention, str) or not mention:
                false_hits += bool(spans)
                continue
            with_mention += 1
            span = _span_of(text, mention)
            n_spans += len(spans)
            hits = [s for s in spans if span and s["start"] < span[1] and span[0] < s["end"]]
            good_spans += len(hits)
            overlap += bool(hits)
            exact += any(normalize(s["text"]) == normalize(mention) for s in spans)
            gold = set(normalize(mention).split())
            got = set(" ".join(normalize(s["text"]) for s in spans).split())
            common = len(gold & got)
            f1s.append(0.0 if not common else 2 * common / (len(gold) + len(got)))
            if name == "combined" and not hits:
                rows.append(
                    {"text": text, "mention": mention, "extracted": [s["text"] for s in spans]}
                )
        without = int((~df.location_mention.fillna("").astype(bool)).sum())
        metrics.update(
            {
                f"location_{name}_exact": exact / max(with_mention, 1),
                f"location_{name}_overlap_recall": overlap / max(with_mention, 1),
                f"location_{name}_token_f1": float(np.mean(f1s)) if f1s else 0.0,
                f"location_{name}_precision": good_spans / max(n_spans, 1),
                f"location_{name}_false_rate": false_hits / max(without, 1),
            }
        )
    return metrics, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- cameras


def camera_matching_metrics(
    df: pd.DataFrame, matcher: CameraMatcher
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """accuracy (None counts as wrong), coverage, precision when a camera was returned, and
    accuracy per source; plus the misses for inspection."""
    results = [matcher.match(t) for t in df.text]
    pred = pd.Series([r.camera_id for r in results], index=df.index)
    correct = pred == df.camera_id
    matched = pred.notna()
    metrics = {
        "camera_accuracy": float(correct.mean()),
        "camera_coverage": float(matched.mean()),
        "camera_precision": float(correct[matched].mean()) if matched.any() else 0.0,
    }
    by_source = (
        pd.DataFrame({"source": df.source, "correct": correct, "matched": matched})
        .groupby("source")
        .agg(n=("correct", "size"), accuracy=("correct", "mean"), coverage=("matched", "mean"))
        .reset_index()
    )
    for r in by_source.itertuples():
        metrics[f"camera_accuracy_{r.source}"] = float(r.accuracy)
    misses = df.loc[~correct, ["text", "source", "camera_id", "location_mention"]].assign(
        predicted=pred[~correct],
        confidence=[results[i].confidence for i in np.flatnonzero(~correct.to_numpy())],
    )
    return metrics, by_source, misses


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
