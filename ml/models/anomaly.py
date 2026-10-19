"""LSTM autoencoder anomaly detector over sliding windows of per-camera features.

Trained on normal periods only. Each feature's reconstruction error over the last
SCORE_STEPS windows (recent behaviour, so alarms fire quickly) is divided by a high
quantile of that feature's error on normal validation data for the camera ("max_scaled",
the default), and the score is the largest of these. That way a break in one smooth channel
(e.g. loop-detector speed) isn't drowned out by noise in the others; "mean" (plain mean
error over features) is kept for comparison. Features are normalised per camera, and
thresholds are per camera (percentile of that camera's normal validation scores) with a
global fallback for unseen cameras. Everything is stored with the model.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import (
    AE_FEATURES,
    GLOBAL,
    LEVEL_FEATURES,
    SEQUENCE_LENGTH,
    CameraReference,
    base_features,
    build_sequences,
    level_distances,
    sequence_segments,
)
from ml.hints import derive_hints

SCORE_STEPS = 3
CLIP = 10.0
_FLAG_FEATURES = {"camera_speed_present", "sensor_present", "hour_sin", "hour_cos"}


@dataclass
class Normalizer:
    """Per-camera mean/std of AE_FEATURES; GLOBAL for unseen cameras."""

    stats: dict[str, dict[str, list[float]]]

    @classmethod
    def fit(cls, base: pd.DataFrame) -> "Normalizer":
        def one(df: pd.DataFrame) -> dict[str, list[float]]:
            mean = df[AE_FEATURES].mean().to_numpy(float)
            std = df[AE_FEATURES].std(ddof=0).fillna(0).to_numpy(float)
            # Floors keep near-constant features (a sensor that never dropped out) from
            # exploding into huge z-scores, while still flagging real changes.
            floor = np.array([0.25 if f in _FLAG_FEATURES else 0.0 for f in AE_FEATURES])
            floor = np.maximum(floor, 0.01 + 0.05 * np.abs(mean))
            return {"mean": mean.tolist(), "std": np.maximum(std, floor).tolist()}

        stats = {str(cam): one(df) for cam, df in base.groupby("camera_id")}
        stats[GLOBAL] = one(base)
        return cls(stats)

    def arrays(self, cameras: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        keys = cameras.map(lambda c: c if c in self.stats else GLOBAL)
        mean = np.stack([self.stats[k]["mean"] for k in keys])
        std = np.stack([self.stats[k]["std"] for k in keys])
        return mean.astype(np.float32), std.astype(np.float32)

    def transform(self, base: pd.DataFrame) -> np.ndarray:
        mean, std = self.arrays(base["camera_id"])
        z = (base[AE_FEATURES].to_numpy(np.float32) - mean) / std
        return np.clip(z, -CLIP, CLIP)


def make_autoencoder(n_features: int, hidden: int = 32, latent: int = 16, layers: int = 1):
    import torch
    from torch import nn

    class LSTMAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.decoder = nn.LSTM(latent, hidden, num_layers=layers, batch_first=True)
            self.output = nn.Linear(hidden, n_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h, _) = self.encoder(x)
            z = self.to_latent(h[-1])
            repeated = z.unsqueeze(1).repeat(1, x.shape[1], 1)
            out, _ = self.decoder(repeated)
            return self.output(out)

    return LSTMAutoencoder()


@dataclass
class AnomalyConfig:
    hidden: int = 32
    latent: int = 16
    layers: int = 1
    sequence_length: int = SEQUENCE_LENGTH
    score_steps: int = SCORE_STEPS
    threshold_percentile: float = 99.5
    score_method: str = "max_scaled"  # or "mean"
    # Each feature's error is scaled by this quantile of its normal error, so spiky
    # (heavy-tailed) features need a truly unusual spike while smooth channels stay sensitive.
    scale_quantile: float = 99.0
    min_threshold_sequences: int = 200  # fewer normal val sequences -> use the global threshold
    # Level term: distance from the (rolling or frozen) baseline per LEVEL_FEATURES column,
    # scaled by this quantile of its normal distance and weighted. 0 disables it.
    level_weight: float = 1.0
    level_quantile: float = 99.0


@dataclass
class Calibration:
    """Per-camera reconstruction-error scales, level scales and thresholds (GLOBAL as
    fallback), fitted on normal validation sequences after training."""

    error_scale: dict[str, list[float]]
    thresholds: dict[str, float]
    level_scale: dict[str, list[float]] = field(default_factory=dict)

    def scales(self, cameras: np.ndarray) -> np.ndarray:
        return np.stack([self.error_scale.get(c, self.error_scale[GLOBAL]) for c in cameras])

    def level_scales(self, cameras: np.ndarray) -> np.ndarray | None:
        if not self.level_scale:
            return None
        return np.stack([self.level_scale.get(c, self.level_scale[GLOBAL]) for c in cameras])

    def threshold(self, camera: str) -> float:
        return self.thresholds.get(camera, self.thresholds[GLOBAL])


def combine(errors: np.ndarray, scales: np.ndarray | None, method: str) -> np.ndarray:
    """Per-feature errors [N, F] -> scores [N]."""
    if method == "mean" or scales is None:
        return errors.mean(axis=1)
    return (errors / scales).max(axis=1)


@dataclass
class ScoreParts:
    reconstruction: np.ndarray  # [N, F] scaled reconstruction error per AE feature
    level: np.ndarray  # [N, L] weighted, scaled distance from baseline per LEVEL feature
    reconstruction_score: np.ndarray  # [N]
    level_score: np.ndarray  # [N]
    score: np.ndarray  # [N] the larger of the two


def score_parts(errors: np.ndarray, levels: np.ndarray | None, cameras: np.ndarray,
                cal: "Calibration", cfg: "AnomalyConfig") -> ScoreParts:  # fmt: skip
    """Combine reconstruction errors [N, F] and level distances [N, L] into scores."""
    scales = cal.scales(cameras) if cfg.score_method != "mean" else None
    recon = errors / scales if scales is not None else errors
    recon_score = combine(errors, scales, cfg.score_method)
    level_scales = cal.level_scales(cameras)
    if levels is None or level_scales is None or cfg.level_weight <= 0:
        level = np.zeros((len(errors), len(LEVEL_FEATURES)), dtype=float)
    else:
        level = cfg.level_weight * levels / level_scales
    level_score = level.max(axis=1) if level.size else np.zeros(len(errors))
    return ScoreParts(recon, level, recon_score, level_score, np.maximum(recon_score, level_score))


def calibrate(errors: np.ndarray, cameras: np.ndarray, cfg: "AnomalyConfig",
              levels: np.ndarray | None = None) -> Calibration:  # fmt: skip
    def scale(values: np.ndarray, q: float, floor: float) -> list[float]:
        return np.maximum(np.percentile(values, q, axis=0), floor).tolist()

    error_scale = {GLOBAL: scale(errors, cfg.scale_quantile, 1e-4)}
    level_scale = {}
    if levels is not None:
        # Floors: a feature that never moved in normal data (e.g. no stopped vehicles at all)
        # must still need a real change (0.1 relative, or 1 vehicle) to count as far from normal.
        floors = np.array([0.1] * (len(LEVEL_FEATURES) - 2) + [1.0, 1.0])
        level_scale[GLOBAL] = np.maximum(scale(levels, cfg.level_quantile, 0.0), floors).tolist()
    for cam in np.unique(cameras):
        mask = cameras == cam
        error_scale[str(cam)] = scale(errors[mask], cfg.scale_quantile, 1e-4)
        if levels is not None:
            level_scale[str(cam)] = np.maximum(
                scale(levels[mask], cfg.level_quantile, 0.0), floors
            ).tolist()
    cal = Calibration(error_scale, {}, level_scale)
    scores = score_parts(errors, levels, cameras, cal, cfg).score
    cal.thresholds[GLOBAL] = float(np.percentile(scores, cfg.threshold_percentile))
    for cam in np.unique(cameras):
        mask = cameras == cam
        if mask.sum() >= cfg.min_threshold_sequences:
            cal.thresholds[str(cam)] = float(np.percentile(scores[mask], cfg.threshold_percentile))
    return cal


def reconstruct(model, seqs: np.ndarray, batch: int = 2048) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            out.append(model(torch.from_numpy(seqs[i : i + batch])).numpy())
    return np.concatenate(out) if out else np.empty_like(seqs)


def sequence_errors(seqs: np.ndarray, recon: np.ndarray, steps: int = SCORE_STEPS) -> np.ndarray:
    """Per-feature squared error averaged over the last ``steps`` windows: [N, F]."""
    return ((seqs[:, -steps:, :] - recon[:, -steps:, :]) ** 2).mean(axis=1)


def train_autoencoder(
    train: np.ndarray,
    val: np.ndarray,
    cfg: AnomalyConfig,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    batch: int = 256,
    seed: int = 0,
    log=None,
):
    import torch

    torch.manual_seed(seed)
    model = make_autoencoder(train.shape[2], cfg.hidden, cfg.latent, cfg.layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    X = torch.from_numpy(train)
    gen = torch.Generator().manual_seed(seed)
    history, best, best_state = [], float("inf"), None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X), generator=gen)
        total = 0.0
        for i in range(0, len(order), batch):
            xb = X[order[i : i + batch]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val_loss = (
            float(((reconstruct(model, val) - val) ** 2).mean()) if len(val) else float("nan")
        )
        history.append({"epoch": epoch, "train_loss": total / len(X), "val_loss": val_loss})
        if log:
            log(history[-1])
        if val_loss < best:
            best, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@dataclass
class ScoredFrame:
    """Scores for every row that ends a full sequence (others are NaN)."""

    base: pd.DataFrame  # base features, sorted by camera, ts
    scores: np.ndarray  # [rows]
    thresholds: np.ndarray  # [rows] the camera's threshold (NaN without calibration)
    errors: np.ndarray  # [rows, F] per-feature error (scaled if calibrated) over the scored steps
    expected: np.ndarray  # [rows, F] reconstruction of the last window, de-normalised
    levels: np.ndarray  # [rows, L] weighted, scaled level distance (0 without calibration)
    reconstruction_scores: np.ndarray  # [rows]
    level_scores: np.ndarray  # [rows]


def window_levels(base: pd.DataFrame, ends: np.ndarray, steps: int) -> np.ndarray:
    """Level distances averaged over the last ``steps`` rows of each sequence: [N, L]."""
    dist = level_distances(base)
    cum = np.vstack([np.zeros((1, dist.shape[1])), np.cumsum(dist, axis=0)])
    return (cum[ends + 1] - cum[ends + 1 - steps]) / steps


def score_frame(
    raw: pd.DataFrame,
    refs: dict[str, CameraReference],
    normalizer: Normalizer,
    model,
    cfg: AnomalyConfig,
    calibration: Calibration | None = None,
) -> ScoredFrame:
    base = base_features(raw, refs)
    z = normalizer.transform(base)
    seqs, ends = build_sequences(z, sequence_segments(base), cfg.sequence_length)
    n, f, lf = len(base), len(AE_FEATURES), len(LEVEL_FEATURES)
    scores, thresholds = np.full(n, np.nan), np.full(n, np.nan)
    recon_scores, level_scores = np.full(n, np.nan), np.full(n, np.nan)
    errors, expected = np.full((n, f), np.nan), np.full((n, f), np.nan)
    levels = np.zeros((n, lf))
    if len(seqs):
        recon = reconstruct(model, seqs)
        err = sequence_errors(seqs, recon, cfg.score_steps)
        cams = base["camera_id"].to_numpy()[ends].astype(str)
        if calibration:
            parts = score_parts(
                err, window_levels(base, ends, cfg.score_steps), cams, calibration, cfg
            )
            scores[ends], errors[ends], levels[ends] = (
                parts.score,
                parts.reconstruction,
                parts.level,
            )
            recon_scores[ends], level_scores[ends] = parts.reconstruction_score, parts.level_score
            thresholds[ends] = [calibration.threshold(c) for c in cams]
        else:
            scores[ends] = recon_scores[ends] = combine(err, None, cfg.score_method)
            errors[ends], level_scores[ends] = err, 0.0
        mean, std = normalizer.arrays(base["camera_id"].iloc[ends])
        expected[ends] = recon[:, -1, :] * std + mean
    return ScoredFrame(
        base, scores, thresholds, errors, expected, levels, recon_scores, level_scores
    )


RECON, LEVEL = "reconstruction", "level"


def explain_row(sf: ScoredFrame, i: int, top: int = 6) -> dict:
    """Contributions (share of all score components, keyed "reconstruction:<feature>" or
    "level:<feature>"), which part fired, the top feature with observed/expected, and hints.

    Level features compare to the baseline: expected is 1.0 for <col>_rel and 0 for excesses.
    """
    components = {f"{RECON}:{f}": float(e) for f, e in zip(AE_FEATURES, sf.errors[i], strict=True)}
    components.update(
        {f"{LEVEL}:{f}": float(v) for f, v in zip(LEVEL_FEATURES, sf.levels[i], strict=True)}
    )
    components = {k: v for k, v in components.items() if np.isfinite(v) and v > 0}
    total = sum(components.values()) or 1.0
    ranked = sorted(components, key=components.get, reverse=True)
    contributions = {k: round(components[k] / total, 4) for k in ranked[:top]}

    observed = {f: float(sf.base[f].iloc[i]) for f in set(AE_FEATURES) | set(LEVEL_FEATURES)}
    expected = {f: float(v) for f, v in zip(AE_FEATURES, sf.expected[i], strict=True)}
    level_expected = {f: (0.0 if f.endswith("_excess") else 1.0) for f in LEVEL_FEATURES}
    fired_by = LEVEL if sf.level_scores[i] > sf.reconstruction_scores[i] else RECON
    part, top_feature = ranked[0].split(":", 1) if ranked else (fired_by, AE_FEATURES[0])
    top_expected = level_expected[top_feature] if part == LEVEL else expected[top_feature]

    by_feature: dict[str, float] = {}  # hints look at features, whichever part they came from
    for key, share in contributions.items():
        feature = key.split(":", 1)[1]
        by_feature[feature] = by_feature.get(feature, 0.0) + share
    hint_expected = {**expected, **{f: v for f, v in level_expected.items() if f not in expected}}
    for f in LEVEL_FEATURES:  # a level contribution always compares against the baseline
        if f"{LEVEL}:{f}" in contributions and f"{RECON}:{f}" not in contributions:
            hint_expected[f] = level_expected[f]
    sensor_missing = bool(sf.base["sensor_present"].iloc[i] == 0)
    return {
        "contributions": contributions,
        "fired_by": fired_by,
        "top_feature": top_feature,
        "observed": observed[top_feature],
        "expected": top_expected,
        "hints": derive_hints(by_feature, observed, hint_expected, sensor_missing),
    }


# --------------------------------------------------------------------------- MLflow pyfunc

try:
    import mlflow.pyfunc

    _PythonModel = mlflow.pyfunc.PythonModel
except ImportError:
    _PythonModel = object


def save_artifacts(
    directory: Path,
    model,
    normalizer: Normalizer,
    refs: dict[str, CameraReference],
    cfg: AnomalyConfig,
    calibration: Calibration,
) -> dict[str, str]:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "autoencoder.pt")
    meta = {
        "features": AE_FEATURES,
        "config": asdict(cfg),
        "calibration": asdict(calibration),
        "normalizer": normalizer.stats,
        "references": {k: v.to_dict() for k, v in refs.items()},
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"weights": str(directory / "autoencoder.pt"), "meta": str(directory / "meta.json")}


class AnomalyPyfunc(_PythonModel):
    """Input: raw rows (features.RAW_COLUMNS), each camera's recent history.
    Output: one row per camera for its latest window: valid (a full contiguous sequence
    was available), score, threshold, is_anomaly, reconstruction_score, level_score,
    fired_by, contributions (JSON), top_feature, observed, expected, hints (JSON).

    Optional input columns baseline_<col> (ml.features.FREEZABLE_COLUMNS) replace the rolling
    baseline where set; anomaly_service passes the baseline it froze when an alert started."""

    def load_context(self, context) -> None:
        import torch

        meta = json.loads(Path(context.artifacts["meta"]).read_text(encoding="utf-8"))
        self.cfg = AnomalyConfig(**meta["config"])
        self.calibration = Calibration(**meta["calibration"])
        self.normalizer = Normalizer(meta["normalizer"])
        self.refs = {k: CameraReference.from_dict(v) for k, v in meta["references"].items()}
        self.model = make_autoencoder(
            len(meta["features"]), self.cfg.hidden, self.cfg.latent, self.cfg.layers
        )
        self.model.load_state_dict(torch.load(context.artifacts["weights"], weights_only=True))
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        sf = score_frame(
            model_input, self.refs, self.normalizer, self.model, self.cfg, self.calibration
        )
        rows = []
        for camera, idx in sf.base.groupby("camera_id", sort=False).indices.items():
            i = int(idx[-1])
            score = sf.scores[i]
            threshold = self.calibration.threshold(str(camera))
            row = {
                "camera_id": sf.base["camera_id"].iloc[i],
                "ts": sf.base["ts"].iloc[i],
                "valid": bool(not np.isnan(score)),
                "threshold": threshold,
            }
            if np.isnan(score):
                row.update(
                    score=np.nan,
                    is_anomaly=False,
                    reconstruction_score=np.nan,
                    level_score=np.nan,
                    fired_by=None,
                    contributions="{}",
                    top_feature=None,
                    observed=np.nan,
                    expected=np.nan,
                    hints="[]",
                )
            else:
                ex = explain_row(sf, i)
                row.update(
                    score=float(score),
                    is_anomaly=bool(score >= threshold),
                    reconstruction_score=float(sf.reconstruction_scores[i]),
                    level_score=float(sf.level_scores[i]),
                    fired_by=ex["fired_by"],
                    contributions=json.dumps(ex["contributions"]),
                    top_feature=ex["top_feature"],
                    observed=ex["observed"],
                    expected=ex["expected"],
                    hints=json.dumps(ex["hints"]),
                )
            rows.append(row)
        return pd.DataFrame(rows)
