"""Helpers shared by the training CLIs."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]  # the ml package, bundled into pyfunc models
log = logging.getLogger("ml.training")


def setup(threads: int = 6) -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    # One INFO line per Hugging Face Hub request (huggingface_hub resets the httpx level).
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda record: not record.name.startswith("httpx"))
    try:
        import torch

        torch.set_num_threads(threads)  # all cores is slower inside the Docker VM
    except ImportError:
        pass
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def time_split(df: pd.DataFrame, fractions: tuple[float, ...]) -> pd.Series:
    """Split label per row by time within each (origin, camera): first fraction is "train",
    then "val", then "test". Time-based so rolling features and sequences don't leak."""
    names = ["train", "val", "test"][: len(fractions)]
    edges = np.cumsum(fractions)
    # Positional (not label-based), so duplicate index labels can't mix cameras up.
    order = df.groupby(["origin", "camera_id"])["ts"].rank(method="first", pct=True).to_numpy()
    labels = np.select([order <= e for e in edges], names, default=names[-1])
    return pd.Series(labels, index=df.index, dtype=object)


def pip_requirements(*files: str) -> list[str]:
    """The model runtime pins, so serving installs exactly what training used."""
    lines = []
    for name in files or ("requirements-runtime.txt",):
        lines += (ML_DIR / name).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith(("#", "-r"))]


def register(run_id: str, name: str, alias: str, description: str, tags: dict[str, str]) -> str:
    """Register runs:/<run_id>/model as a new version of ``name`` and point ``alias`` at it."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mv = mlflow.register_model(f"runs:/{run_id}/model", name, tags=tags)
    client = MlflowClient()
    client.update_model_version(name, mv.version, description=description)
    client.set_registered_model_alias(name, alias, mv.version)
    log.info("registered %s version %s and set @%s", name, mv.version, alias)
    return str(mv.version)
