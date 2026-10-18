"""Report text embeddings for embedding drift.

With a DistilBERT classifier, a report's embedding is the mean of its encoder's last hidden
states (the representation the classifier itself works from). With the TF-IDF baseline it is
a hashed character n-gram vector. Either way a PCA fitted on the reference reduces it to
``EMBEDDING_DIMS`` dimensions; the PCA is stored with the reference so current reports are
projected the same way."""

import numpy as np
import pandas as pd

from ml.monitoring.specs import EMBEDDING_COLUMNS, EMBEDDING_DIMS


class TextEmbedder:
    def __init__(self, classifier) -> None:
        """``classifier``: the unwrapped IncidentTextClassifier (ml.nlp.pyfunc)."""
        self.classifier = classifier
        self.kind = (
            "encoder_mean" if getattr(classifier, "kind", None) == "transformer" else "hashing"
        )

    def raw(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.kind == "encoder_mean":
            import torch

            from ml.nlp.transformer import encode

            model, tokenizer, cfg = (
                self.classifier.model,
                self.classifier.tokenizer,
                self.classifier.cfg,
            )
            model.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    enc = encode(tokenizer, texts[i : i + batch_size], cfg.max_length)
                    hidden = model.encoder(**enc).last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    out.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy())
            return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float32)
        from sklearn.feature_extraction.text import HashingVectorizer

        from ml.nlp.labels import preprocess

        vec = HashingVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), n_features=2**12, alternate_sign=False
        )
        return vec.transform([preprocess(t) for t in texts]).toarray().astype(np.float32)


def fit_pca(vectors: np.ndarray, dims: int = EMBEDDING_DIMS) -> dict[str, np.ndarray]:
    mean = vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(vectors - mean, full_matrices=False)
    return {"mean": mean.astype(np.float32), "components": vt[:dims].astype(np.float32)}


def project(vectors: np.ndarray, pca: dict[str, np.ndarray]) -> pd.DataFrame:
    z = (vectors - pca["mean"]) @ pca["components"].T
    return pd.DataFrame(z, columns=list(EMBEDDING_COLUMNS[: z.shape[1]]))
