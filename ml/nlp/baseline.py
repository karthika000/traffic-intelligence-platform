"""TF-IDF + logistic regression baseline: word 1-2 grams plus character 2-5 grams (robust to
typos and abbreviations), one shared vectoriser and one classifier per task."""

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml.nlp.labels import TASKS, preprocess


class TfidfBaseline:
    def __init__(self, c: float = 10.0, max_iter: int = 3000, seed: int = 0) -> None:
        self.c, self.max_iter, self.seed = c, max_iter, seed
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2
        )
        self.models: dict[str, LogisticRegression] = {}

    def params(self) -> dict:
        return {
            "baseline_C": self.c,
            "baseline_word_ngrams": "1-2",
            "baseline_char_ngrams": "2-5 (char_wb)",
        }

    def _features(self, texts: list[str], fit: bool = False):
        clean = [preprocess(t) for t in texts]
        if fit:
            return hstack([self.word.fit_transform(clean), self.char.fit_transform(clean)]).tocsr()
        return hstack([self.word.transform(clean), self.char.transform(clean)]).tocsr()

    def fit(self, texts: list[str], labels: dict[str, np.ndarray]) -> "TfidfBaseline":
        x = self._features(texts, fit=True)
        for task in TASKS:
            model = LogisticRegression(C=self.c, max_iter=self.max_iter, random_state=self.seed)
            self.models[task] = model.fit(x, labels[task])
        return self

    def predict_proba(self, texts: list[str]) -> dict[str, np.ndarray]:
        x = self._features(texts)
        out = {}
        for task, model in self.models.items():
            proba = model.predict_proba(x)
            # Classes missing from training would shift columns: place them by class id.
            full = np.zeros((x.shape[0], int(model.classes_.max()) + 1))
            full[:, model.classes_] = proba
            out[task] = full
        return out
