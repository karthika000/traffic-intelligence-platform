"""Generate a labelled NLP dataset of traffic reports.

    python -m event_simulator.dataset [--out DIR] [--train 5000] [--test 1000]

Writes reports_train.csv and reports_test.csv with columns
text, incident_type, severity, camera_id, location_mention, source.

The test split uses a different seed and the holdout phrase pools (unseen templates,
incident and severity phrases, location forms) about half the time, so a good score
on it means the model generalises rather than memorising templates. Exact duplicate
texts are removed within each split, and any test text also present in train is dropped.

In Docker, the default output /data/nlp is ./data/nlp on the host.
"""

import argparse
import csv
import random
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from event_simulator.config import SimulatorConfig
from event_simulator.locations import CameraInfo
from event_simulator.reports import (
    INCIDENT_WEIGHTS_DATASET,
    GeneratedReport,
    ReportGenerator,
    Stage,
    Variant,
)
from traffic_common.db.seed import DEMO_CAMERAS
from traffic_common.enums import IncidentType

COLUMNS = ["text", "incident_type", "severity", "camera_id", "location_mention", "source"]
CLEARED_SHARE = 0.08  # share of incident reports describing a cleared incident
TRAIN_SEED, TEST_SEED = 20261014, 777


def demo_cameras() -> list[CameraInfo]:
    return [
        CameraInfo(c["id"], c["name"], c["location_name"], c["latitude"], c["longitude"])
        for c in DEMO_CAMERAS
    ]


def generate_split(
    cameras: list[CameraInfo],
    size: int,
    seed: int,
    variant: Variant,
    exclude: set[str] = frozenset(),
) -> list[GeneratedReport]:
    rng = random.Random(seed)
    gen = ReportGenerator(rng, variant)
    rows: list[GeneratedReport] = []
    seen = set(exclude)
    attempts = 0
    while len(rows) < size:
        attempts += 1
        if attempts > size * 50:
            raise RuntimeError(f"could only generate {len(rows)} distinct reports")
        incident_type = gen.sample_incident_type(INCIDENT_WEIGHTS_DATASET)
        stage = (
            Stage.CLEARED
            if incident_type is not IncidentType.NONE and rng.random() < CLEARED_SHARE
            else Stage.ACTIVE
        )
        report = gen.generate(rng.choice(cameras), incident_type=incident_type, stage=stage)
        if report.text in seen:
            continue
        seen.add(report.text)
        rows.append(report)
    return rows


def write_csv(path: Path, rows: Iterable[GeneratedReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow(
                [
                    r.text,
                    r.incident_type.value,
                    r.severity.value,
                    r.camera_id,
                    r.location_mention,
                    r.style.value,
                ]
            )


def distribution_table(name: str, rows: list[GeneratedReport]) -> str:
    lines = [f"{name}: {len(rows)} rows"]
    for title, key in (("incident_type", lambda r: r.incident_type.value),
                       ("severity", lambda r: r.severity.value),
                       ("source", lambda r: r.style.value)):  # fmt: skip
        counts = Counter(key(r) for r in rows)
        lines.append(f"  {title}:")
        for label, n in counts.most_common():
            lines.append(f"    {label:<16} {n:5d}  {100 * n / len(rows):5.1f}%")
    with_location = sum(1 for r in rows if r.location_mention)
    lines.append(f"  with a location mention: {100 * with_location / len(rows):.1f}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m event_simulator.dataset",
                                     description=__doc__.split("\n\n")[0])  # fmt: skip
    parser.add_argument("--out", type=Path, default=SimulatorConfig.from_env().dataset_dir)
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--test-seed", type=int, default=TEST_SEED)
    args = parser.parse_args(argv)

    cameras = demo_cameras()
    train = generate_split(cameras, args.train, args.train_seed, Variant.TRAIN)
    test = generate_split(cameras, args.test, args.test_seed, Variant.TEST,
                          exclude={r.text for r in train})  # fmt: skip

    train_path, test_path = args.out / "reports_train.csv", args.out / "reports_test.csv"
    write_csv(train_path, train)
    write_csv(test_path, test)
    print(f"wrote {train_path} ({len(train)} rows) and {test_path} ({len(test)} rows)\n")
    print(distribution_table("train", train))
    print()
    print(distribution_table("test", test))
    return 0


if __name__ == "__main__":
    sys.exit(main())
