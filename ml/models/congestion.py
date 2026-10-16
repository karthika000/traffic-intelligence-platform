"""Congestion classifier: gradient boosting baseline and a small PyTorch MLP on engineered
window features, plus the MLflow pyfunc wrapper (raw rows in, level per camera out)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import CONGESTION_FEATURES, CameraReference, base_features, congestion_features
from ml.labels import CONGESTION_LEVELS


def feature_matrix(raw: pd.DataFrame, refs: dict[str, CameraReference]) -> pd.DataFrame:
    """Engineered features (CONGESTION_FEATURES), row-aligned with ``raw`` sorted by camera, ts."""
    feats = congestion_features(base_features(raw, refs))
    return feats


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(
            y_true, y_pred, labels=range(len(CONGESTION_LEVELS)), average="macro", zero_division=0
        )
    )


# --------------------------------------------------------------------------- gradient boosting


def train_gbm(X: np.ndarray, y: np.ndarray, seed: int = 0, max_iter: int = 300):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        class_weight="balanced",
        random_state=seed,
    )
    return model.fit(X, y)


# --------------------------------------------------------------------------- MLP


class MLPClassifier:
    """A small MLP with input standardisation; sklearn-like predict / predict_proba."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.1):
        import torch.nn as nn

        layers, width = [], n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, len(CONGESTION_LEVELS)))
        self.net = nn.Sequential(*layers)
        self.hidden, self.dropout = hidden, dropout
        self.mean = np.zeros(n_features, dtype=np.float32)
        self.std = np.ones(n_features, dtype=np.float32)

    def _scaled(self, X: np.ndarray):
        import torch

        return torch.from_numpy(
            ((np.asarray(X, np.float32) - self.mean) / self.std).astype(np.float32)
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(self._scaled(X)), dim=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def fit(
        self,
        X,
        y,
        X_val,
        y_val,
        *,
        epochs: int = 40,
        lr: float = 1e-3,
        batch: int = 512,
        patience: int = 6,
        seed: int = 0,
    ) -> list[dict]:
        import torch

        torch.manual_seed(seed)
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = np.maximum(X.std(axis=0), 1e-6).astype(np.float32)
        counts = np.bincount(y, minlength=len(CONGESTION_LEVELS)).astype(np.float32)
        weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(counts))
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        Xt, yt = self._scaled(X), torch.from_numpy(np.asarray(y, np.int64))
        best, best_state, history, since_best = -1.0, None, [], 0
        gen = torch.Generator().manual_seed(seed)
        for epoch in range(epochs):
            self.net.train()
            order = torch.randperm(len(Xt), generator=gen)
            total = 0.0
            for i in range(0, len(order), batch):
                idx = order[i : i + batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            f1 = macro_f1(y_val, self.predict(X_val))
            history.append({"epoch": epoch, "train_loss": total / len(Xt), "val_macro_f1": f1})
            if f1 > best:
                best, since_best = f1, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                since_best += 1
                if since_best >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return history

    def save(self, path: Path) -> None:
        import torch

        torch.save(
            {
                "state": self.net.state_dict(),
                "mean": self.mean,
                "std": self.std,
                "hidden": self.hidden,
                "dropout": self.dropout,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, n_features: int) -> "MLPClassifier":
        import torch

        blob = torch.load(path, weights_only=False)
        model = cls(n_features, tuple(blob["hidden"]), blob["dropout"])
        model.net.load_state_dict(blob["state"])
        model.mean, model.std = blob["mean"], blob["std"]
        return model


# --------------------------------------------------------------------------- evaluation helpers


def permutation_importance(
    model, X: np.ndarray, y: np.ndarray, names: list[str], repeats: int = 3, seed: int = 0
) -> pd.DataFrame:
    """Drop in macro-F1 when each feature is shuffled (same method for both model types)."""
    rng = np.random.default_rng(seed)
    base = macro_f1(y, model.predict(X))
    rows = []
    for j, name in enumerate(names):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - macro_f1(y, model.predict(Xp)))
        rows.append(
            {"feature": name, "importance": float(np.mean(drops)), "std": float(np.std(drops))}
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- MLflow pyfunc

try:
    import mlflow.pyfunc

    _PythonModel = mlflow.pyfunc.PythonModel
except ImportError:  # the feature/label code is importable without MLflow
    _PythonModel = object


class CongestionPyfunc(_PythonModel):
    """Input: raw rows (features.RAW_COLUMNS), the recent history of one or more cameras.
    Output: one row per camera for its latest window: level, confidence, proba_<level>."""

    def load_context(self, context) -> None:
        meta = json.loads(Path(context.artifacts["meta"]).read_text(encoding="utf-8"))
        self.kind = meta["kind"]
        self.features = meta["features"]
        self.refs = {k: CameraReference.from_dict(v) for k, v in meta["references"].items()}
        if self.kind == "gbm":
            import joblib

            self.model = joblib.load(context.artifacts["model"])
        else:
            self.model = MLPClassifier.load(Path(context.artifacts["model"]), len(self.features))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        feats = feature_matrix(model_input, self.refs)
        last = feats.groupby("camera_id", sort=False).tail(1)
        proba = self.model.predict_proba(last[self.features].to_numpy(np.float32))
        out = pd.DataFrame(
            {
                "camera_id": last["camera_id"].to_numpy(),
                "ts": last["ts"].to_numpy(),
                "level": np.asarray(CONGESTION_LEVELS)[proba.argmax(axis=1)],
                "confidence": proba.max(axis=1),
            }
        )
        for i, level in enumerate(CONGESTION_LEVELS):
            out[f"proba_{level}"] = proba[:, i]
        return out


__all__ = [
    "CONGESTION_FEATURES",
    "CongestionPyfunc",
    "MLPClassifier",
    "feature_matrix",
    "train_gbm",
]
