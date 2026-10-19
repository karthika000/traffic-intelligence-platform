"""Model tests that need torch (run in the Docker test image): the autoencoder and both
congestion models round-trip through MLflow pyfunc packaging and score raw rows."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="torch only in the Docker images")
pytest.importorskip("mlflow")
import mlflow.pyfunc  # noqa: E402
import pandas as pd  # noqa: E402

from ml.data import synthetic  # noqa: E402
from ml.features import (  # noqa: E402
    AE_FEATURES,
    CONGESTION_FEATURES,
    LEVEL_FEATURES,
    RAW_COLUMNS,
    base_features,
    compute_references,
)  # noqa: E402
from ml.labels import CONGESTION_LEVELS  # noqa: E402
from ml.models.anomaly import (  # noqa: E402
    AnomalyConfig,
    AnomalyPyfunc,
    Normalizer,
    calibrate,
    reconstruct,
    save_artifacts,
    sequence_errors,
    train_autoencoder,
    window_levels,
)
from ml.models.congestion import CongestionPyfunc, MLPClassifier, feature_matrix, train_gbm  # noqa: E402
from ml.training.train_anomaly import normal_sequences  # noqa: E402

ML_DIR = ROOT / "ml"


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return synthetic.generate(cameras=2, days=0.3, seed=5, episodes_per_day=12)


def test_autoencoder_pyfunc_round_trip(data: pd.DataFrame, tmp_path: Path) -> None:
    refs = compute_references(data[data.is_anomaly == 0])
    base = base_features(data, refs)
    normalizer = Normalizer.fit(base[base.is_anomaly == 0])
    seqs, cams, ends = normal_sequences(base, normalizer.transform(base), stride=3)
    cfg = AnomalyConfig(hidden=8, latent=4, threshold_percentile=99.0, min_threshold_sequences=50)
    model, history = train_autoencoder(seqs, seqs[:200], cfg, epochs=2)
    assert len(history) == 2 and history[-1]["train_loss"] < history[0]["train_loss"] * 1.5
    levels = window_levels(base, ends, cfg.score_steps)
    calibration = calibrate(sequence_errors(seqs, reconstruct(model, seqs)), cams, cfg, levels)
    assert set(calibration.thresholds) >= {"__global__", "syn-cam-1", "syn-cam-2"}

    artifacts = save_artifacts(tmp_path / "artifacts", model, normalizer, refs, cfg, calibration)
    mlflow.pyfunc.save_model(
        str(tmp_path / "model"),
        python_model=AnomalyPyfunc(),
        artifacts=artifacts,
        code_paths=[str(ML_DIR)],
    )
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))

    recent = data.groupby("camera_id").tail(20)[RAW_COLUMNS]
    out = loaded.predict(recent)
    assert list(out["camera_id"]) == ["syn-cam-1", "syn-cam-2"]
    assert out["valid"].all() and (out["score"] >= 0).all()
    assert out.loc[0, "threshold"] == pytest.approx(calibration.thresholds["syn-cam-1"])
    contributions = json.loads(out.loc[0, "contributions"])
    parts = {key.split(":", 1)[0] for key in contributions}
    features = {key.split(":", 1)[1] for key in contributions}
    assert parts <= {"reconstruction", "level"} and features <= set(AE_FEATURES) | set(
        LEVEL_FEATURES
    )
    assert sum(contributions.values()) <= 1.0001
    assert out.loc[0, "fired_by"] in ("reconstruction", "level")
    assert out.loc[0, "score"] == pytest.approx(
        max(out.loc[0, "reconstruction_score"], out.loc[0, "level_score"])
    )

    # A frozen baseline far above the current loop speed raises the level part.
    frozen = recent.assign(baseline_loop_speed_kmh=recent["loop_speed_kmh"].max() * 5)
    assert loaded.predict(frozen).loc[0, "level_score"] > out.loc[0, "level_score"]
    assert isinstance(json.loads(out.loc[0, "hints"]), list)

    short = loaded.predict(data.groupby("camera_id").tail(5)[RAW_COLUMNS])
    assert not short["valid"].any()  # fewer than 12 windows: no sequence to score
    unseen = loaded.predict(recent.assign(camera_id="cam-unseen").iloc[-15:])
    assert unseen.loc[0, "threshold"] == pytest.approx(calibration.thresholds["__global__"])


@pytest.mark.parametrize("kind", ["gbm", "mlp"])
def test_congestion_pyfunc_round_trip(data: pd.DataFrame, tmp_path: Path, kind: str) -> None:
    import joblib

    refs = compute_references(data)
    feats = feature_matrix(data, refs)
    X = feats[CONGESTION_FEATURES].to_numpy(np.float32)
    y = feats["congestion"].map({lvl: i for i, lvl in enumerate(CONGESTION_LEVELS)}).to_numpy()
    if kind == "gbm":
        model = train_gbm(X, y, max_iter=30)
        path = tmp_path / "model.joblib"
        joblib.dump(model, path)
    else:
        model = MLPClassifier(len(CONGESTION_FEATURES), hidden=(16,))
        model.fit(X, y, X[:500], y[:500], epochs=2)
        path = tmp_path / "model.pt"
        model.save(path)
    assert (model.predict(X) == y).mean() > 0.6
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "kind": kind,
                "features": CONGESTION_FEATURES,
                "levels": CONGESTION_LEVELS,
                "references": {k: v.to_dict() for k, v in refs.items()},
            }
        )
    )
    mlflow.pyfunc.save_model(
        str(tmp_path / "pyfunc"),
        python_model=CongestionPyfunc(),
        artifacts={"model": str(path), "meta": str(meta)},
        code_paths=[str(ML_DIR)],
    )
    out = mlflow.pyfunc.load_model(str(tmp_path / "pyfunc")).predict(
        data.groupby("camera_id").tail(15)[RAW_COLUMNS]
    )
    assert list(out["camera_id"]) == ["syn-cam-1", "syn-cam-2"]
    assert set(out["level"]) <= set(CONGESTION_LEVELS)
    probas = out[[f"proba_{lvl}" for lvl in CONGESTION_LEVELS]].to_numpy()
    assert np.allclose(probas.sum(axis=1), 1, atol=1e-4)
    assert np.allclose(out["confidence"], probas.max(axis=1))


def test_camera_calibration_fills_what_the_model_lacks(data: pd.DataFrame, tmp_path: Path) -> None:
    """anomaly_service.calibration: an unseen camera gets its own video scales (never below the
    global ones) and threshold from its warm-up; loop features keep the model's scales; ignored
    signals never score; a calibrated camera keeps its own values."""
    import copy

    sys.path.insert(0, str(ROOT / "services" / "anomaly_service"))
    from anomaly_service.calibration import CameraSettings, calibrate_camera

    refs = compute_references(data[data.is_anomaly == 0])
    base = base_features(data, refs)
    normalizer = Normalizer.fit(base[base.is_anomaly == 0])
    seqs, cams, ends = normal_sequences(base, normalizer.transform(base), stride=3)
    cfg = AnomalyConfig(hidden=8, latent=4, min_threshold_sequences=50)
    model, _ = train_autoencoder(seqs, seqs[:200], cfg, epochs=1)
    levels = window_levels(base, ends, cfg.score_steps)
    calibration = calibrate(sequence_errors(seqs, reconstruct(model, seqs)), cams, cfg, levels)
    py = AnomalyPyfunc()
    py.cfg, py.calibration, py.normalizer, py.refs, py.model = (
        cfg,
        calibration,
        normalizer,
        refs,
        model,
    )
    pristine = copy.deepcopy(calibration)
    normal = data[(data.camera_id == "syn-cam-1") & (data.is_anomaly == 0)][RAW_COLUMNS]
    # An unseen camera whose video speeds are far noisier than any training camera's.
    rng = np.random.default_rng(0)
    noisy = normal.iloc[:200].assign(camera_id="cam-noisy")
    noisy["avg_speed_kmh"] *= rng.choice([0.05, 1.0, 20.0], len(noisy))

    result = calibrate_camera(py, pristine, "cam-noisy", noisy, CameraSettings())
    assert result is not None and not result.own_scales and not result.own_threshold
    g, own = pristine.error_scale["__global__"], calibration.error_scale["cam-noisy"]
    i_speed, i_loop = (
        AE_FEATURES.index("avg_speed_kmh_rel"),
        AE_FEATURES.index("loop_speed_kmh_rel"),
    )
    assert own[i_speed] > g[i_speed]  # its own, noisier scale
    assert own[i_loop] == g[i_loop]  # loop features keep the model's scale
    assert all(o >= x for o, x in zip(own, g, strict=True))
    assert result.threshold >= pristine.thresholds["__global__"]
    assert calibration.threshold("cam-noisy") == result.threshold
    assert "cam-noisy" not in pristine.error_scale  # the pristine copy is untouched

    too_short = calibrate_camera(py, pristine, "cam-new", noisy.iloc[:30], CameraSettings())
    assert too_short is None  # not enough sequences: the warm-up goes on

    ignore = CameraSettings(ignore=("camera_speed",), threshold_factor=2.0)
    result = calibrate_camera(py, pristine, "cam-noisy", noisy, ignore)
    assert calibration.error_scale["cam-noisy"][i_speed] == float("inf")
    out = py.predict(None, noisy.iloc[-20:])
    contributions = json.loads(out.loc[0, "contributions"])
    assert not any("avg_speed_kmh_rel" in k or "p85_speed_kmh_rel" in k for k in contributions)

    own_threshold = pristine.thresholds["syn-cam-1"]
    result = calibrate_camera(py, pristine, "syn-cam-1", normal.iloc[:5], ignore)
    assert result.own_scales and result.own_threshold  # nothing fitted, so no history needed
    assert result.threshold == pytest.approx(2 * own_threshold)
    assert CameraSettings.from_zone_config(
        {"anomaly": {"ignore": ["camera_speed", "bogus"], "threshold_factor": 1.5}}
    ) == CameraSettings(("camera_speed",), 1.5)
