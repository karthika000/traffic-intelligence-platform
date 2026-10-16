"""Pretrained Hugging Face NER (default dslim/bert-base-NER: PER, ORG, LOC, MISC), loaded
lazily. Downloads go to HF_HOME (a named Docker volume), so restarts don't re-download."""

import logging
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_NER_MODEL = "dslim/bert-base-NER"


class NerExtractor:
    def __init__(
        self, model_name: str = DEFAULT_NER_MODEL, min_score: float = 0.5, batch_size: int = 16
    ) -> None:
        self.model_name, self.min_score, self.batch_size = model_name, min_score, batch_size
        self._pipe = None

    @property
    def version(self) -> str:
        """The Hub commit the weights came from, when known."""
        self.load()
        commit = getattr(self._pipe.model.config, "_commit_hash", None)
        if not commit:  # transformers 5 doesn't keep it: the cache snapshot dir is the commit
            try:
                from huggingface_hub import try_to_load_from_cache

                cached = try_to_load_from_cache(self.model_name, "config.json")
                commit = Path(cached).parent.name if isinstance(cached, str) else None
            except Exception:
                commit = None
        return commit[:12] if commit else "unknown"

    def load(self) -> "NerExtractor":
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
                device=-1,
            )
            log.info("NER model %s loaded", self.model_name)
        return self

    def extract(self, texts: list[str]) -> list[list[dict]]:
        """Per text: [{"text", "label" (PER/ORG/LOC/MISC), "start", "end", "score"}]."""
        if not texts:
            return []
        self.load()
        results = self._pipe(list(texts), batch_size=self.batch_size)
        if texts and isinstance(results, list) and results and isinstance(results[0], dict):
            results = [results]  # a single text comes back unnested
        out = []
        for text, ents in zip(texts, results, strict=True):
            spans = []
            for e in ents:
                if float(e["score"]) < self.min_score:
                    continue
                snapped = snap_to_words(text, int(e["start"]), int(e["end"]))
                if snapped is None:
                    continue
                start, end = snapped
                spans.append(
                    {
                        "text": text[start:end],
                        "label": e["entity_group"],
                        "start": start,
                        "end": end,
                        "score": float(e["score"]),
                    }
                )
            out.append(spans)
        return out


def snap_to_words(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim a span that starts or ends inside a word back to whole words ("Faraday Memorial
    AG" from "...Memorial AGAIN" -> "Faraday Memorial"); None if no whole word is left (the
    subword model sometimes tags a fragment such as "GO" in "GONE")."""
    if end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end = text.rfind(" ", start, end)
    if 0 < start < end and text[start - 1].isalnum() and text[start].isalnum():
        nxt = text.find(" ", start, end)
        start = nxt + 1 if nxt >= 0 else end
    while start < end and not text[end - 1].isalnum():
        end -= 1
    return (start, end) if end > start else None
