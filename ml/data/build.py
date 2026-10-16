"""Build a versioned training dataset (real + synthetic) as parquet under data/ml/.

    python -m ml.data.build [--real-hours 24 | --no-real] [--synthetic-days 2] [--seed 0]

Every row keeps its origin ("real" or "synthetic") and label source ("weak" for real
congestion labels, "synthetic" for generated ground truth). The version is
<UTC timestamp>-<first 8 hex of the content hash>; the full SHA-256 is in stats.json.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml.data import real as real_data
from ml.data import synthetic
from ml.features import RAW_COLUMNS

log = logging.getLogger(__name__)
DATA_ROOT = Path("data/ml")
LABEL_COLUMNS = [
    "is_anomaly",
    "anomaly_type",
    "episode_id",
    "congestion",
    "origin",
    "label_source",
    "scenario_id",
]


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    version: str
    path: Path
    sha256: str
    stats: dict

    def mlflow_params(self) -> dict[str, str | int]:
        return {
            "dataset_name": self.name,
            "dataset_version": self.version,
            "dataset_sha256": self.sha256,
            "dataset_rows_real": self.stats["rows_by_origin"].get("real", 0),
            "dataset_rows_synthetic": self.stats["rows_by_origin"].get("synthetic", 0),
        }

    def log_to_mlflow(self) -> None:
        """Log the dataset version, hash and stats to the active run. (Not mlflow.log_input:
        it profiles and hashes every row again, which took ~9 minutes for 200k rows.)"""
        import mlflow

        mlflow.log_params(self.mlflow_params())
        mlflow.log_dict(self.stats, "dataset/stats.json")
        mlflow.set_tags({"dataset_path": str(self.path), "dataset_sha256": self.sha256})


def content_hash(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df.astype({"scenario_id": str}), index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def build_dataset(
    name: str,
    *,
    engine=None,
    include_real: bool = True,
    real_hours: float | None = None,
    synthetic_days: float = 2.0,
    synthetic_cameras: int = 4,
    seed: int = 0,
    root: Path = DATA_ROOT,
) -> tuple[pd.DataFrame, DatasetVersion, pd.DataFrame]:
    """Returns (rows, version, scenario windows)."""
    parts, windows = [], pd.DataFrame()
    if include_real:
        if engine is None:
            from traffic_common.db import create_db_engine

            engine = create_db_engine()
        real, windows = real_data.load_real(engine, real_hours)
        log.info("real data: %s", real_data.summarise(real))
        if not real.empty:
            parts.append(real[RAW_COLUMNS + LABEL_COLUMNS])
    if synthetic_days > 0:
        syn = synthetic.generate(synthetic_cameras, synthetic_days, seed)
        parts.append(syn[RAW_COLUMNS + LABEL_COLUMNS])
    if not parts:
        raise ValueError("no data: include real data or synthetic days")
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["origin", "camera_id", "ts"]).reset_index(drop=True)

    digest = content_hash(df)
    version = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{digest[:8]}"
    stats = {
        "rows": len(df),
        "rows_by_origin": df["origin"].value_counts().to_dict(),
        "cameras_by_origin": df.groupby("origin")["camera_id"].nunique().to_dict(),
        "time_range_by_origin": {
            o: [str(g["ts"].min()), str(g["ts"].max())] for o, g in df.groupby("origin")
        },
        "congestion_by_origin": {
            o: g["congestion"].value_counts().to_dict() for o, g in df.groupby("origin")
        },
        "anomaly_rows_by_type": df.loc[df["is_anomaly"] == 1, "anomaly_type"]
        .value_counts()
        .to_dict(),
        "anomaly_episodes": int(df.loc[df["episode_id"] >= 0, "episode_id"].nunique()),
        "scenario_runs": int(len(windows)),
        "sensor_coverage_by_origin": df.groupby("origin")["loop_speed_kmh"]
        .apply(lambda s: round(float(s.notna().mean()), 3))
        .to_dict(),
        "settings": {
            "include_real": include_real,
            "real_hours": real_hours,
            "synthetic_days": synthetic_days,
            "synthetic_cameras": synthetic_cameras,
            "seed": seed,
        },
        "sha256": digest,
    }
    path = root / name / version
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / "dataset.parquet", index=False)
    if not windows.empty:
        windows.to_parquet(path / "scenario_windows.parquet", index=False)
    (path / "stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log.info(
        "dataset %s/%s: %d rows %s, sha256 %s",
        name,
        version,
        len(df),
        stats["rows_by_origin"],
        digest[:12],
    )
    return df, DatasetVersion(name, version, path, digest, stats), windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="traffic")
    parser.add_argument("--real-hours", type=float, default=None, help="default: all real data")
    parser.add_argument("--no-real", action="store_true", help="synthetic data only")
    parser.add_argument("--synthetic-days", type=float, default=2.0)
    parser.add_argument("--synthetic-cameras", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, version, _ = build_dataset(
        args.name,
        include_real=not args.no_real,
        real_hours=args.real_hours,
        synthetic_days=args.synthetic_days,
        synthetic_cameras=args.synthetic_cameras,
        seed=args.seed,
    )
    print(json.dumps(version.stats, indent=2, default=str))


if __name__ == "__main__":
    main()
