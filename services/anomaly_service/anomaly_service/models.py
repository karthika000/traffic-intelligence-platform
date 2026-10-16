"""Champion models from the MLflow registry, with rule-based fallbacks and hot reload."""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from ml.models.rules import RuleBasedAnomaly, RuleBasedCongestion

log = logging.getLogger(__name__)


class Predictor(Protocol):
    def predict(self, raw: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class LoadedModel:
    role: str  # "congestion" | "anomaly"
    name: str
    version: str  # registry version, or "rules-1" for the fallback
    predictor: Predictor
    source: str  # "mlflow" | "rules"
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.name}/{self.version}"


Loader = Callable[[str], Predictor]  # model URI -> pyfunc model


def _mlflow_loader(uri: str) -> Predictor:
    import mlflow.pyfunc

    return mlflow.pyfunc.load_model(uri)


class ModelManager:
    """Keeps the champion of each role loaded.

    ``refresh()`` asks the registry which version each alias points at and reloads when it
    changed. Without a registered champion the role falls back to rules (logged at WARNING);
    if MLflow is unreachable the currently loaded model is kept.
    """

    def __init__(
        self,
        tracking_uri: str,
        roles: dict[str, str],  # role -> registered model name
        alias: str = "champion",
        *,
        client: Any = None,
        loader: Loader = _mlflow_loader,
        on_change: Callable[[LoadedModel], None] | None = None,
    ) -> None:
        self.roles = roles
        self.alias = alias
        self._client = client
        self._tracking_uri = tracking_uri
        self._loader = loader
        self._on_change = on_change
        self._lock = threading.Lock()
        self.models: dict[str, LoadedModel] = {}

    def get(self, role: str) -> LoadedModel:
        with self._lock:
            return self.models[role]

    @property
    def client(self) -> Any:
        if self._client is None:
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient(self._tracking_uri)
        return self._client

    def refresh(self) -> list[LoadedModel]:
        """Load whatever changed; returns the models that were (re)loaded."""
        changed = []
        for role, name in self.roles.items():
            try:
                model = self._resolve(role, name)
            except Exception as exc:  # registry unreachable, broken model, ...
                current = self.models.get(role)
                if current is None:
                    log.warning("%s: MLflow unavailable (%s); using rule-based fallback", role, exc)
                    model = self._rules(role)
                else:
                    log.warning(
                        "%s: model refresh failed (%s); keeping %s", role, exc, current.label
                    )
                    continue
            if model is None:
                continue
            with self._lock:
                self.models[role] = model
            changed.append(model)
            if self._on_change:
                self._on_change(model)
        return changed

    def _resolve(self, role: str, name: str) -> LoadedModel | None:
        """The model to switch to, or None if the current one is still the champion."""
        current = self.models.get(role)
        version = self._champion_version(name)
        if version is None:
            if current is None or current.source != "rules":
                log.warning(
                    "%s: no %s@%s registered in MLflow; using the RULE-BASED fallback detector",
                    role,
                    name,
                    self.alias,
                )
                return self._rules(role)
            return None
        if current is not None and current.source == "mlflow" and current.version == version:
            return None
        uri = f"models:/{name}/{version}"
        predictor = self._loader(uri)
        mv = self.client.get_model_version(name, version)
        metadata = getattr(getattr(predictor, "metadata", None), "metadata", None) or {}
        log.info(
            "%s: loaded %s v%s (run %s)%s",
            role,
            name,
            version,
            mv.run_id,
            f", replacing {current.label}" if current else "",
        )
        return LoadedModel(role, name, version, predictor, "mlflow", mv.run_id, metadata)

    def _champion_version(self, name: str) -> str | None:
        from mlflow.exceptions import MlflowException

        try:
            registered = self.client.get_registered_model(name)
        except MlflowException as exc:
            if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
                return None
            raise
        aliases = registered.aliases or {}
        version = aliases.get(self.alias)
        return str(version) if version is not None else None

    @staticmethod
    def _rules(role: str) -> LoadedModel:
        predictor = RuleBasedCongestion() if role == "congestion" else RuleBasedAnomaly()
        return LoadedModel(role, predictor.name, predictor.version, predictor, "rules")
