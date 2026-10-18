"""Reference snapshots: what each registered model version's inputs and outputs look like on
the data it was trained (or validated) on. Drift checks compare production with them.

    python -m ml.monitoring.reference build <model> [--version N] [--minutes 30]
    python -m ml.monitoring.reference build-all          # every champion without one
    python -m ml.monitoring.reference show <model> [--version N]

Each snapshot is ``data/ml/reference/<model>/v<version>/`` with ``reference.parquet`` (one row
per sample, the columns of the model's check in ml.monitoring.specs), ``meta.json`` (source,
rows, settings) and, for the text model, ``pca.npz``. It is also logged to the model version's
MLflow run under ``reference/`` (tag ``reference_path``).

Where each reference comes from:

- **congestion-classifier, traffic-anomaly-detector:** the real rows of the training dataset
  (synthetic ones only if there are too few real rows), with the model's own predictions on
  them, computed by the registered model.
- **incident-text-classifier:** reports_train.csv, resampled to the incident mix of recent
  production reports (simulator ground truth, standard style) because the training set is
  deliberately flattened; predictions, camera matching and encoder embeddings are computed on
  it. Without production history it keeps the training mix.
- **vehicle-detector:** it is pretrained, not trained here, so its reference is a window of
  its own output on the clean camera feeds (the ``detections`` topic, read back by time) plus
  brightness and contrast of the current frame files. The window must not overlap drift mode.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.monitoring.data import (
    RAW_INPUTS,
    anomaly_outputs,
    detection_rows,
    frame_stats,
    read_detections,
    text_features,
)

log = logging.getLogger(__name__)
ROOT = Path("data/ml/reference")
MAX_ROWS = 20_000
MIN_REAL_ROWS = 2_000
# vehicle-detector: about 2.5 minutes of the four demo cameras at 5 fps. On a fresh install
# ml-monitor's first check comes before cv_inference has produced much; refusing then means the
# next check builds a real reference instead of saving a near-empty one for good.
MIN_DETECTIONS = 3_000
TEXT_ROWS = 3_000
NLP_TRAIN = Path("data/nlp/reports_train.csv")


@dataclass
class Reference:
    model_name: str
    version: str
    frame: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    path: Path | None = None


def reference_dir(model_name: str, version: str, root: Path = ROOT) -> Path:
    return root / model_name / f"v{version}"


def save(ref: Reference, root: Path = ROOT) -> Path:
    path = reference_dir(ref.model_name, ref.version, root)
    path.mkdir(parents=True, exist_ok=True)
    frame = ref.frame.copy()
    for c in frame.columns:  # parquet needs one type per column
        if frame[c].dtype == object:
            frame[c] = frame[c].map(lambda v: None if pd.isna(v) else str(v))
    frame.to_parquet(path / "reference.parquet", index=False)
    meta = {
        **ref.meta,
        "model_name": ref.model_name,
        "model_version": ref.version,
        "rows": len(ref.frame),
        "columns": list(ref.frame.columns),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if ref.arrays:
        np.savez(path / "pca.npz", **ref.arrays)
    ref.path, ref.meta = path, meta
    log.info("reference %s v%s: %d rows -> %s", ref.model_name, ref.version, len(frame), path)
    return path


def load(model_name: str, version: str, root: Path = ROOT) -> Reference | None:
    path = reference_dir(model_name, version, root)
    if not (path / "reference.parquet").exists():
        return None
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(path / "pca.npz")) if (path / "pca.npz").exists() else {}
    return Reference(model_name, version, pd.read_parquet(path / "reference.parquet"), meta,
                     arrays, path)  # fmt: skip


def log_to_mlflow(ref: Reference, run_id: str | None, client=None) -> None:
    """Attach the snapshot to the model version's run (artifacts under reference/)."""
    if not run_id or ref.path is None:
        return
    from mlflow.tracking import MlflowClient

    client = client or MlflowClient()
    try:
        client.log_artifacts(run_id, str(ref.path), artifact_path="reference")
        client.set_tag(run_id, "reference_path", str(ref.path))
        client.set_tag(run_id, "reference_rows", str(len(ref.frame)))
    except Exception as exc:  # the snapshot on disk is what the checks use
        log.warning("could not log reference to MLflow run %s: %s", run_id, exc)


def champion(model_name: str, client=None, alias: str = "champion") -> tuple[str, str | None]:
    """(version, run_id) of the model's alias in the MLflow registry."""
    from mlflow.tracking import MlflowClient

    mv = (client or MlflowClient()).get_model_version_by_alias(model_name, alias)
    return str(mv.version), mv.run_id


def _sample(frame: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return frame.sample(n, random_state=seed) if len(frame) > n else frame


# ----------------------------------------------------------------------------- traffic models


def dataset_for_run(run_id: str, client=None) -> pd.DataFrame:
    """The training dataset a run logged (params dataset_name / dataset_version)."""
    from mlflow.tracking import MlflowClient

    params = (client or MlflowClient()).get_run(run_id).data.params
    path = Path("data/ml") / params["dataset_name"] / params["dataset_version"] / "dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"training dataset {path} (run {run_id}) is not on disk")
    return pd.read_parquet(path)


def reference_rows(dataset: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rows from the camera pipeline itself (origin real), or all rows if too few are real.
    Sorted by camera and time, as the models' feature code expects."""
    real = dataset[dataset["origin"] == "real"]
    rows, source = (real, "real") if len(real) >= MIN_REAL_ROWS else (dataset, "real+synthetic")
    rows = rows.copy()
    rows["ts"] = pd.to_datetime(rows["ts"], utc=True)
    return rows.sort_values(["camera_id", "ts"]).reset_index(drop=True), source


def congestion_outputs(python_model, raw: pd.DataFrame) -> pd.DataFrame:
    """Level and confidence for every row (raw sorted by camera, ts)."""
    from ml.models.congestion import feature_matrix

    feats = feature_matrix(raw, python_model.refs)
    proba = python_model.model.predict_proba(feats[python_model.features].to_numpy(np.float32))
    from ml.labels import CONGESTION_LEVELS

    return pd.DataFrame(
        {
            "predicted_level": np.asarray(CONGESTION_LEVELS)[proba.argmax(axis=1)],
            "prediction_confidence": proba.max(axis=1),
        }
    )


def anomaly_scores(python_model, raw: pd.DataFrame) -> pd.DataFrame:
    """Score / threshold and above-threshold for every row (NaN before a full sequence)."""
    from ml.models.anomaly import score_frame

    sf = score_frame(
        raw,
        python_model.refs,
        python_model.normalizer,
        python_model.model,
        python_model.cfg,
        python_model.calibration,
    )
    return anomaly_outputs(pd.Series(sf.scores), pd.Series(sf.thresholds))


def build_traffic(
    model_name: str, version: str, python_model, dataset: pd.DataFrame, seed: int = 0
) -> Reference:
    raw, source = reference_rows(dataset)
    keep = raw[["camera_id", "ts", *RAW_INPUTS]].reset_index(drop=True)
    if model_name == "congestion-classifier":
        outputs = congestion_outputs(python_model, keep)
    else:
        outputs = anomaly_scores(python_model, keep)
    frame = pd.concat([keep, outputs.reset_index(drop=True)], axis=1)
    frame = _sample(frame, MAX_ROWS, seed).sort_values(["camera_id", "ts"]).reset_index(drop=True)
    meta = {
        "source": f"training dataset ({source} rows)",
        "time_range": [str(raw["ts"].min()), str(raw["ts"].max())],
        "cameras": sorted(raw["camera_id"].unique().tolist()),
    }
    return Reference(model_name, version, frame, meta)


# ----------------------------------------------------------------------------- text classifier

_PRIOR_SQL = """
SELECT incident_type, count(*) AS n FROM simulated_reports
WHERE emitted_at >= :since AND text_style IS NULL GROUP BY 1
"""


def production_prior(engine, hours: float = 24.0) -> dict[str, float] | None:
    """Incident mix of recent standard-style production reports (simulator ground truth)."""
    if engine is None:
        return None
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(_PRIOR_SQL), {"since": datetime.now(UTC) - timedelta(hours=hours)}
            ).all()
    except Exception as exc:
        log.warning("no production label mix (%s); keeping the training mix", exc)
        return None
    total = sum(r.n for r in rows)
    if total < 200:
        return None
    return {r.incident_type: r.n / total for r in rows}


def resample_to_prior(
    train: pd.DataFrame, prior: dict[str, float] | None, n: int, seed: int = 0
) -> pd.DataFrame:
    if not prior:
        return _sample(train, n, seed)
    parts = []
    for label, share in prior.items():
        rows = train[train["incident_type"] == label]
        k = int(round(n * share))
        if len(rows) and k:
            parts.append(rows.sample(k, replace=k > len(rows), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def text_outputs(classifier, texts: list[str]) -> pd.DataFrame:
    from ml.nlp.predict import prediction_frame

    pred = prediction_frame(classifier.predict_probabilities(texts), classifier.maps)
    return pd.DataFrame(
        {
            "predicted_incident_type": pred["incident_type"],
            "predicted_severity": pred["severity"],
            "incident_confidence": pred["incident_confidence"],
            "severity_confidence": pred["severity_confidence"],
        }
    )


def default_matcher():
    from ml.nlp.matching import CameraMatcher
    from traffic_common.db.seed import DEMO_CAMERAS
    from traffic_common.gazetteer import CameraInfo

    return CameraMatcher(
        [
            CameraInfo(c["id"], c["name"], c["location_name"], c["latitude"], c["longitude"])
            for c in DEMO_CAMERAS
        ]
    )


def extra_training_reports(run_id: str | None, client) -> pd.DataFrame | None:
    """Reports a retrained text model was trained on besides reports_train.csv (train_nlp's
    --extra-train, a parameter of the candidate's parent run), if the file is still there."""
    if not run_id:
        return None
    try:
        run = client.get_run(run_id)
        parent = run.data.tags.get("mlflow.parentRunId")
        params = client.get_run(parent).data.params if parent else run.data.params
    except Exception:
        return None
    path = Path(params.get("extra_train") or "")
    if not params.get("extra_train") or not path.is_file():
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def build_text(
    version: str,
    classifier,
    train: pd.DataFrame,
    engine=None,
    n: int = TEXT_ROWS,
    seed: int = 0,
) -> Reference:
    from ml.monitoring.embeddings import TextEmbedder, fit_pca, project

    prior = production_prior(engine)
    rows = resample_to_prior(train, prior, n, seed).reset_index(drop=True)
    texts = rows["text"].astype(str).tolist()
    matcher = default_matcher()
    matched = pd.Series([matcher.match(t).camera_id for t in texts])
    embedder = TextEmbedder(classifier)
    vectors = embedder.raw(texts)
    pca = fit_pca(vectors)
    frame = pd.concat(
        [
            rows[["text"]].reset_index(drop=True),
            text_outputs(classifier, texts),
            text_features(rows["text"], matched),
            project(vectors, pca),
        ],
        axis=1,
    )
    meta = {
        "source": "reports_train.csv"
        + (" resampled to the production incident mix" if prior else ""),
        "label_prior": prior or train["incident_type"].value_counts(normalize=True).to_dict(),
        "embedding": embedder.kind,
    }
    return Reference("incident-text-classifier", version, frame, meta, pca)


# ----------------------------------------------------------------------------- vehicle detector


def build_detector(
    version: str,
    bootstrap: str,
    engine=None,
    minutes: float = 30.0,
    frames_dir: Path = Path("/data/frames"),
    allow_drift: bool = False,
) -> Reference:
    from traffic_common.driftmode import active_periods, read_state

    end = datetime.now(UTC)
    start = end - timedelta(minutes=minutes)
    if engine is not None and not allow_drift:
        if active_periods(engine, start, end) or read_state(engine).active():
            raise RuntimeError(
                "drift mode was on in the reference window; switch it off and wait, or pass "
                "--minutes to use a shorter clean window"
            )
    detections = read_detections(bootstrap, start, end)
    if len(detections) < MIN_DETECTIONS:
        raise RuntimeError(
            f"only {len(detections)} detections in the last {minutes:g} min (at least "
            f"{MIN_DETECTIONS} needed): let cv_inference run for longer first"
        )
    frame = detection_rows(detections, version)
    if frame.empty:
        frame = detection_rows(detections)  # e.g. a version label cv_inference didn't publish
    stats = frame_stats(frames_dir, start, max_files=800)
    frame = pd.concat([_sample(frame, MAX_ROWS), stats], ignore_index=True)
    meta = {
        "source": f"detections topic, {start:%Y-%m-%d %H:%M}-{end:%H:%M} UTC (clean feeds)",
        "messages": len(detections),
        "frame_files": len(stats),
    }
    return Reference("vehicle-detector", version, frame, meta)


# ----------------------------------------------------------------------------- dispatch


def build(
    model_name: str,
    version: str | None = None,
    *,
    engine=None,
    client=None,
    bootstrap: str | None = None,
    minutes: float = 30.0,
    frames_dir: Path = Path("/data/frames"),
    root: Path = ROOT,
    allow_drift: bool = False,
) -> Reference:
    """Build, save and log the reference for ``model_name`` at ``version`` (the champion by
    default)."""
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient

    client = client or MlflowClient()
    if version is None:
        version, run_id = champion(model_name, client)
    else:
        run_id = client.get_model_version(model_name, version).run_id
    if model_name == "vehicle-detector":
        ref = build_detector(
            version,
            bootstrap or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            engine,
            minutes,
            frames_dir,
            allow_drift,
        )
    else:
        python_model = mlflow.pyfunc.load_model(
            f"models:/{model_name}/{version}"
        ).unwrap_python_model()
        if model_name == "incident-text-classifier":
            train = pd.read_csv(NLP_TRAIN, keep_default_na=False, na_values=[""])
            extra = extra_training_reports(run_id, client)
            if extra is not None:
                train = pd.concat([train, extra], ignore_index=True)
            ref = build_text(version, python_model, train, engine)
            if extra is not None:
                ref.meta["source"] += (
                    f", plus {len(extra)} recent labelled reports it was trained on"
                )
        else:
            ref = build_traffic(model_name, version, python_model, dataset_for_run(run_id, client))
    ref.meta["mlflow_run_id"] = run_id
    save(ref, root)
    log_to_mlflow(ref, run_id, client)
    return ref


def save_for_run(model_name: str, version: str, run_id: str, **kwargs) -> Reference | None:
    """Training hook: build the new version's reference right after it is registered. A
    failure is logged, not raised: the model is already registered, and ml-monitor builds a
    missing reference itself."""
    try:
        return build(model_name, version, **kwargs)
    except Exception as exc:
        log.warning("could not build the reference for %s v%s: %s", model_name, version, exc)
        return None


def main() -> None:
    from ml.monitoring.config import MONITORED_MODELS

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build (or rebuild) one model's reference")
    b.add_argument("model", choices=MONITORED_MODELS)
    b.add_argument("--version", default=None, help="registry version (default: the champion)")
    b.add_argument("--minutes", type=float, default=30.0, help="vehicle-detector window")
    b.add_argument("--allow-drift", action="store_true", help="vehicle-detector: skip the check")
    sub.add_parser("build-all", help="build every champion's reference that is missing")
    s = sub.add_parser("show", help="print a reference's metadata and column summary")
    s.add_argument("model", choices=MONITORED_MODELS)
    s.add_argument("--version", default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    import mlflow

    from traffic_common.config import get_settings
    from traffic_common.db import create_db_engine

    settings = get_settings("ml-monitor")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    engine = create_db_engine(settings.postgres_url)
    frames_dir = Path(os.getenv("FRAMES_DIR", "/data/frames"))
    if args.command == "show":
        version = args.version or champion(args.model)[0]
        ref = load(args.model, version)
        if ref is None:
            raise SystemExit(f"no reference for {args.model} v{version}")
        print(json.dumps(ref.meta, indent=2, default=str))
        print(ref.frame.describe(include="all").T.to_string())
        return
    models = [args.model] if args.command == "build" else list(MONITORED_MODELS)
    for model in models:
        if args.command == "build-all":
            try:
                version, _ = champion(model)
            except Exception as exc:
                log.warning("%s: no champion (%s)", model, exc)
                continue
            if load(model, version) is not None:
                log.info("%s v%s: reference exists", model, version)
                continue
        ref = build(
            model,
            getattr(args, "version", None),
            engine=engine,
            bootstrap=settings.kafka_bootstrap_servers,
            minutes=getattr(args, "minutes", 30.0),
            frames_dir=frames_dir,
            allow_drift=getattr(args, "allow_drift", False),
        )
        print(json.dumps({k: v for k, v in ref.meta.items() if k != "label_prior"}, default=str))


if __name__ == "__main__":
    main()
