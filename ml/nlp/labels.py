"""Label maps and text preprocessing shared by every incident text classifier."""

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from traffic_common.enums import IncidentType, Severity

TASKS = ("incident_type", "severity")


@dataclass(frozen=True)
class LabelMap:
    """Fixed label order for one task (the enum's order), so ids mean the same thing in every
    model, artifact and evaluation."""

    task: str
    labels: tuple[str, ...]
    to_id: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "to_id", {label: i for i, label in enumerate(self.labels)})

    def __len__(self) -> int:
        return len(self.labels)

    def encode(self, values) -> np.ndarray:
        unknown = sorted({v for v in values if v not in self.to_id})
        if unknown:
            raise ValueError(f"{self.task}: unknown labels {unknown}")
        return np.array([self.to_id[v] for v in values], dtype=np.int64)

    def decode(self, ids) -> list[str]:
        return [self.labels[int(i)] for i in ids]

    def to_dict(self) -> dict:
        return {"task": self.task, "labels": list(self.labels)}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelMap":
        return cls(d["task"], tuple(d["labels"]))


def label_maps() -> dict[str, LabelMap]:
    return {
        "incident_type": LabelMap("incident_type", tuple(t.value for t in IncidentType)),
        "severity": LabelMap("severity", tuple(s.value for s in Severity)),
    }


_URL = re.compile(r"https?://\S+|www\.\S+")
_HASHTAG = re.compile(r"#(\w+)")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPACE = re.compile(r"\s+")


def split_hashtag(tag: str) -> str:
    """Split a CamelCase hashtag: HydeParkCorner -> Hyde Park Corner (lowercase tags stay)."""
    return _CAMEL.sub(" ", tag)


def preprocess(text: str) -> str:
    """Normalise a report for the classifiers: Unicode NFKC, URLs replaced by a placeholder,
    hashtags split into words ("#HydeParkCorner" -> "Hyde Park Corner"), whitespace
    collapsed. Case is kept (the uncased tokenizer and TF-IDF lowercase themselves)."""
    text = unicodedata.normalize("NFKC", str(text))
    text = _URL.sub(" url ", text)
    text = _HASHTAG.sub(lambda m: split_hashtag(m.group(1)), text)
    return _SPACE.sub(" ", text).strip()
