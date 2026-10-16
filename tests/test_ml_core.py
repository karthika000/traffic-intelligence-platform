"""Unit tests for the ml package parts that need no torch: features, labels, windows,
synthetic data, threshold calibration, alarms, hints and the rule-based models."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pd = pytest.importorskip("pandas")
from ml.alarm import AlarmEvent, AlarmTracker, replay  # noqa: E402
from ml.data import synthetic  # noqa: E402
from ml.data.build import content_hash  # noqa: E402
from ml.data.real import label_scenarios, scenario_windows  # noqa: E402
from ml.evaluation.anomaly_eval import episode_metrics, truth_episodes  # noqa: E402
from ml.features import (  # noqa: E402
    AE_FEATURES,
    CONGESTION_FEATURES,
    GLOBAL,
    LEVEL_FEATURES,
    RAW_COLUMNS,
    attach_sensor_readings,
    base_features,
    build_sequences,
    compute_references,
    congestion_features,
    current_baselines,
    level_distances,
    sequence_segments,
)
from ml.hints import derive_hints  # noqa: E402
from ml.labels import CONGESTION_LEVELS, congestion_level, weak_congestion_labels  # noqa: E402
from ml.models.anomaly import AnomalyConfig, calibrate, combine, score_parts  # noqa: E402
from ml.models.rules import RuleBasedAnomaly, RuleBasedCongestion  # noqa: E402
from ml.training.common import time_split  # noqa: E402

T0 = pd.Timestamp("2026-10-15T08:00:00Z")


def rows(n: int, camera: str = "cam-a", start: pd.Timestamp = T0, step_s: float = 5.0, **values):
    """n raw rows every step_s seconds with sensible normal values, overridable per column."""
    base = {
        "vehicle_count": 6,
        "avg_speed_kmh": 30.0,
        "p85_speed_kmh": 38.0,
        "density": 2.0,
        "stopped_vehicles": 0,
        "pedestrians_in_roadway": 0,
        "mean_confidence": 0.6,
        "frames": 24,
        "loop_vehicle_count": 5.0,
        "loop_speed_kmh": 40.0,
        "loop_occupancy": 0.15,
    }
    base.update(values)
    ts = [start + pd.Timedelta(seconds=step_s * i) for i in range(n)]
    return pd.DataFrame({"camera_id": camera, "ts": ts, **base})[RAW_COLUMNS]


# ----------------------------------------------------------------------------- features


def test_attach_sensor_readings_is_backward_asof_with_tolerance() -> None:
    metrics = rows(4).drop(columns=["loop_vehicle_count", "loop_speed_kmh", "loop_occupancy"])
    readings = pd.DataFrame(
        {
            "camera_id": ["cam-a", "cam-a", "cam-b"],
            "reading_at": [T0 - pd.Timedelta(seconds=2), T0 + pd.Timedelta(seconds=11), T0],
            "vehicle_count": [3, 7, 99],
            "avg_speed_kmh": [41.0, 12.0, 99.0],
            "occupancy": [0.1, 0.5, 0.9],
        }
    )
    joined = attach_sensor_readings(metrics, readings, tolerance_s=12)
    # rows at +0, +5, +10 use the reading from -2 s; +15 uses the one from +11 s; never cam-b's
    assert joined["loop_speed_kmh"].tolist() == [41.0, 41.0, 41.0, 12.0]
    stale = attach_sensor_readings(metrics, readings.iloc[:1], tolerance_s=6)
    assert stale["loop_speed_kmh"].isna().tolist() == [False, True, True, True]  # 7 s > 6 s


def test_base_features_fill_flag_and_ratios() -> None:
    raw = rows(6)
    raw.loc[2, "avg_speed_kmh"] = np.nan
    raw.loc[3, ["loop_vehicle_count", "loop_speed_kmh", "loop_occupancy"]] = np.nan
    refs = compute_references(rows(20, avg_speed_kmh=40.0, loop_speed_kmh=50.0))
    base = base_features(raw, {"cam-a": refs["cam-a"]})
    assert base["camera_speed_present"].tolist() == [1, 1, 0, 1, 1, 1]
    assert base["sensor_present"].tolist() == [1, 1, 1, 0, 1, 1]
    assert base.loc[2, "avg_speed_kmh"] == 40.0  # filled with the camera's median
    assert base.loc[0, "speed_ratio"] == pytest.approx(30 / 40)
    assert base.loc[0, "loop_speed_ratio"] == pytest.approx(40 / 50)
    assert base[AE_FEATURES].notna().all().all()
    assert (base["loop_speed_kmh_rel"] == 1.0).all()  # too little history for a baseline yet
    assert {"hour_sin", "hour_cos"} <= set(base.columns)


def test_relative_features_use_a_lagged_baseline() -> None:
    raw = pd.concat(
        [
            rows(60),
            rows(10, start=T0 + pd.Timedelta(seconds=300), loop_speed_kmh=10.0, vehicle_count=0),
        ]
    )
    base = base_features(raw, compute_references(raw))
    rel = base["loop_speed_kmh_rel"]
    assert rel.iloc[59] == pytest.approx(1.0)  # steady traffic
    assert rel.iloc[-1] == pytest.approx(0.25)  # 10 vs a baseline of 40: the drop is visible
    assert base["vehicle_count_rel"].iloc[-1] == pytest.approx(1 / 7)  # (0+1) / (6+1)
    assert np.isfinite(base[AE_FEATURES].to_numpy()).all()


def test_unknown_camera_uses_global_reference() -> None:
    refs = compute_references(pd.concat([rows(10, "cam-a"), rows(10, "cam-b", avg_speed_kmh=60.0)]))
    base = base_features(rows(3, "cam-new", avg_speed_kmh=np.nan), refs)
    assert base["avg_speed_kmh"].iloc[0] == refs[GLOBAL].medians["avg_speed_kmh"]


def test_congestion_features_rolling_per_camera() -> None:
    raw = pd.concat([rows(15, "cam-a"), rows(15, "cam-b", avg_speed_kmh=10.0)])
    feats = congestion_features(base_features(raw, compute_references(raw)))
    assert set(CONGESTION_FEATURES) <= set(feats.columns)
    assert feats[CONGESTION_FEATURES].notna().all().all()
    a, b = feats[feats.camera_id == "cam-a"], feats[feats.camera_id == "cam-b"]
    assert a["speed_ratio_mean_12"].iloc[-1] == pytest.approx(a["speed_ratio"].iloc[-1])
    assert b["speed_ratio_std_12"].iloc[-1] == pytest.approx(0)  # no leakage between cameras


# ----------------------------------------------------------------------------- windows


def test_sequences_respect_camera_and_time_gaps() -> None:
    gap_start = T0 + pd.Timedelta(minutes=10)
    raw = pd.concat([rows(15, "cam-a"), rows(13, "cam-a", start=gap_start), rows(11, "cam-b")])
    base = base_features(raw, compute_references(raw))
    segments = sequence_segments(base)
    assert len(np.unique(segments)) == 3  # cam-a before the gap, after it, cam-b
    values = np.arange(len(base), dtype=np.float32)[:, None]
    seqs, ends = build_sequences(values, segments, length=12)
    assert seqs.shape == (4 + 2, 12, 1)  # 15-12+1 and 13-12+1; cam-b has only 11 rows
    for seq, end in zip(seqs, ends, strict=True):
        assert seq[-1, 0] == end and np.all(np.diff(seq[:, 0]) == 1)  # contiguous rows
    strided, _ = build_sequences(values, segments, length=12, stride=2)
    assert len(strided) == 2 + 1


# ----------------------------------------------------------------------------- labels


@pytest.mark.parametrize(
    ("ratio", "density", "stopped", "level"),
    [
        (0.9, 1.0, 0, "free_flow"),
        (0.6, 1.0, 0, "moderate"),
        (0.9, 1.7, 0, "moderate"),
        (0.3, 1.0, 0, "heavy"),
        (0.55, 2.5, 0, "heavy"),
        (0.1, 1.0, 0, "standstill"),
        (0.35, 1.0, 4, "standstill"),
        (0.9, 1.0, 5, "free_flow"),
    ],
)
def test_congestion_level_thresholds(ratio, density, stopped, level) -> None:
    assert CONGESTION_LEVELS[int(congestion_level(ratio, density, stopped))] == level


def test_weak_labels_prefer_loop_speed() -> None:
    raw = rows(4, avg_speed_kmh=30.0, loop_speed_kmh=5.0)
    raw.loc[3, ["loop_vehicle_count", "loop_speed_kmh", "loop_occupancy"]] = np.nan
    refs = {"cam-a": compute_references(rows(10, avg_speed_kmh=30.0, loop_speed_kmh=50.0))["cam-a"]}
    labels = weak_congestion_labels(base_features(raw, refs))
    assert labels.tolist()[:3] == ["standstill"] * 3  # loop says crawling
    assert labels.iloc[3] == "free_flow"  # no loop reading: camera speed (30 of 30)


def test_scenario_labels() -> None:
    raw = rows(40)
    timeline = [
        {"phase": "a", "stage": "active", "ends_at": str(T0 + pd.Timedelta(seconds=100))},
        {"phase": "b", "stage": "cleared", "ends_at": str(T0 + pd.Timedelta(seconds=150))},
    ]
    runs = pd.DataFrame(
        [
            {
                "id": "s1",
                "scenario_name": "flooding",
                "camera_id": "cam-a",
                "status": "completed",
                "time_scale": 1.0,
                "started_at": T0 + pd.Timedelta(seconds=20),
                "ended_at": T0 + pd.Timedelta(seconds=150),
                "timeline": timeline,
            }
        ]
    )
    labelled = label_scenarios(raw, scenario_windows(runs))
    assert labelled["is_anomaly"].tolist() == [0] * 4 + [1] * 16 + [-1] * 10 + [0] * 10
    assert set(labelled.loc[labelled.is_anomaly == 1, "anomaly_type"]) == {"scenario:flooding"}


# ----------------------------------------------------------------------------- synthetic data


def test_synthetic_data_has_labelled_episodes() -> None:
    df = synthetic.generate(cameras=3, days=1.0, seed=3, episodes_per_day=12)
    assert set(RAW_COLUMNS) <= set(df.columns) and (df["origin"] == "synthetic").all()
    eps = truth_episodes(df)
    assert set(eps["type"]) == set(synthetic.EPISODE_TYPES)
    assert (df.loc[df.is_anomaly == 1, "episode_id"] >= 0).all()
    assert (df.loc[df.is_anomaly == 0, "anomaly_type"] == "").all()
    assert set(df["congestion"]) == set(CONGESTION_LEVELS)
    dropout = df[df.anomaly_type == "sensor_dropout"]
    assert dropout["loop_speed_kmh"].isna().all()
    failure = df[df.anomaly_type == "camera_failure"]
    assert failure["mean_confidence"].mean() < 0.2
    incident = df[df.anomaly_type == "incident"]
    assert incident["stopped_vehicles"].mean() > 2
    assert content_hash(df) == content_hash(df.copy())  # deterministic version hash


def test_time_split_is_per_camera_and_ordered() -> None:
    df = pd.concat([rows(100, "cam-a"), rows(50, "cam-b")]).assign(origin="real")
    split = time_split(df, (0.6, 0.2, 0.2))
    for _, g in df.assign(split=split).groupby("camera_id"):
        assert g["split"].value_counts(normalize=True).round(1).to_dict() == {
            "train": 0.6,
            "val": 0.2,
            "test": 0.2,
        }
        assert g.loc[g.split == "train", "ts"].max() < g.loc[g.split == "test", "ts"].min()


# ----------------------------------------------------------------------------- thresholds


def test_calibration_per_camera_with_global_fallback() -> None:
    rng = np.random.default_rng(0)
    quiet = rng.exponential(1.0, (1000, 3))
    noisy = rng.exponential(5.0, (1000, 3))
    few = rng.exponential(1.0, (50, 3))
    errors = np.vstack([quiet, noisy, few])
    cams = np.array(["quiet"] * 1000 + ["noisy"] * 1000 + ["few"] * 50)
    cfg = AnomalyConfig(threshold_percentile=99.0, min_threshold_sequences=200)
    cal = calibrate(errors, cams, cfg)
    assert set(cal.thresholds) == {GLOBAL, "quiet", "noisy"}  # "few" lacks data: global
    assert cal.threshold("few") == cal.thresholds[GLOBAL] == cal.threshold("never-seen")
    scores = combine(errors[:1000], cal.scales(cams[:1000]), "max_scaled")
    assert (scores >= cal.thresholds["quiet"]).mean() == pytest.approx(0.01, abs=0.002)
    # Scaling by each camera's typical error makes noisy and quiet cameras comparable.
    assert cal.thresholds["noisy"] == pytest.approx(cal.thresholds["quiet"], rel=0.3)
    # A single smooth channel breaking (3x its own p99 error) trips the threshold even though
    # the other channels look normal.
    scales = cal.scales(np.array(["quiet"]))[0]
    one_channel = np.array([[3 * scales[0], scales[1] * 0.3, scales[2] * 0.3]])
    assert combine(one_channel, scales[None, :], "max_scaled")[0] >= cal.thresholds["quiet"]


# ----------------------------------------------------------------------------- alarms


def test_alarm_start_end_and_cooldown() -> None:
    tracker = AlarmTracker(cooldown_s=60, clear_ratio=0.8, clear_after=3, min_consecutive=1)
    thr = 1.0
    events = {}
    series = [0.5, 1.2, 1.5, 0.9, 0.7, 0.6, 0.5, 2.0, 2.0, 0.1, 0.1, 0.1, 2.0]
    for i, s in enumerate(series):
        e = tracker.update(i * 10.0, s, thr)
        if e:
            events[i] = e
    # start at t=10; 0.9 >= 0.8*thr doesn't count as clear; ends after 3 clears (t=60);
    # 2.0 at t=70 and 80 is inside the 60 s cooldown; the next start at t=120 is allowed
    assert events == {1: AlarmEvent.START, 6: AlarmEvent.END, 12: AlarmEvent.START}
    assert tracker.active and tracker.started_at == 120.0


def test_alarm_peak_and_replay() -> None:
    eps = replay(
        [0, 5, 10, 15, 20, 25, 30],
        [0.1, 1.1, 3.0, 0.2, 0.2, 0.2, 0.1],
        1.0,
        cooldown_s=0,
        min_consecutive=1,
    )
    assert len(eps) == 1 and eps[0].start == 5 and eps[0].end == 25 and eps[0].peak == 3.0
    ongoing = replay([0, 5], [2.0, 2.0], 1.0)
    assert ongoing[0].end is None and ongoing[0].start == 0  # start = first crossing


def test_alarm_needs_consecutive_crossings() -> None:
    tracker = AlarmTracker(min_consecutive=2)
    assert tracker.update(0, 5.0, 1.0) is None  # a single spike...
    assert tracker.update(5, 0.2, 1.0) is None  # ...followed by normal: no alarm
    assert tracker.update(10, 1.5, 1.0) is None
    assert tracker.update(15, 1.2, 1.0) is AlarmEvent.START
    assert tracker.started_at == 10 and tracker.peak_score == 1.5


def test_episode_metrics() -> None:
    ts = [T0 + pd.Timedelta(seconds=5 * i) for i in range(400)]
    labels = pd.DataFrame(
        {"camera_id": "c", "ts": ts, "is_anomaly": 0, "episode_id": -1, "anomaly_type": ""}
    )
    labels.loc[100:139, ["is_anomaly", "episode_id", "anomaly_type"]] = (1, 7, "incident")
    labels.loc[300:319, ["is_anomaly", "episode_id", "anomaly_type"]] = (1, 8, "sensor_dropout")
    score = np.zeros(400)
    score[104:140] = 3.0  # detected 20 s in
    score[200:204] = 3.0  # a false alarm
    scored = labels.assign(score=score)
    m, by_type, _ = episode_metrics(
        scored, labels, 1.0, cooldown_s=0, clear_after=1, min_consecutive=1
    )
    assert m["episodes"] == 2 and m["alarms"] == 2
    assert m["recall"] == 0.5 and m["precision"] == 0.5
    assert m["delay_median_s"] == 20.0 and m["false_alarms"] == 1
    assert m["false_alarms_per_hour"] == pytest.approx(1 / (340 * 5 / 3600))
    assert by_type.set_index("type").loc["sensor_dropout", "recall"] == 0.0


# ----------------------------------------------------------------------------- hints and rules


def test_hints_follow_direction_and_share() -> None:
    contributions = {"loop_speed_kmh": 0.6, "stopped_vehicles": 0.3, "vehicle_count": 0.05}
    observed = {"loop_speed_kmh": 6.0, "stopped_vehicles": 4.0, "vehicle_count": 1.0}
    expected = {"loop_speed_kmh": 40.0, "stopped_vehicles": 0.0, "vehicle_count": 6.0}
    assert derive_hints(contributions, observed, expected) == ["speed_drop", "stopped_vehicles"]
    faster = {**observed, "loop_speed_kmh": 70.0}  # above expected: not a speed drop
    assert derive_hints(contributions, faster, expected) == ["stopped_vehicles"]
    assert derive_hints({}, {}, {}, sensor_missing=True) == ["sensor_dropout"]


def test_rule_based_models() -> None:
    normal = rows(30)
    incident = pd.concat(
        [
            normal,
            rows(
                4,
                start=T0 + pd.Timedelta(seconds=150),
                avg_speed_kmh=5.0,
                loop_speed_kmh=5.0,
                stopped_vehicles=4,
                density=6.0,
            ),
        ]
    )
    detector = RuleBasedAnomaly()
    quiet = detector.predict(normal).iloc[0]
    hit = detector.predict(incident).iloc[0]
    assert not quiet["is_anomaly"] and quiet["score"] < 1.0
    assert hit["is_anomaly"] and hit["score"] >= 1.0
    assert json.loads(hit["hints"])[:2] == [
        "speed_drop",
        "stopped_vehicles",
    ] or "speed_drop" in json.loads(hit["hints"])
    dropout = pd.concat(
        [
            normal,
            rows(
                4,
                start=T0 + pd.Timedelta(seconds=150),
                loop_speed_kmh=np.nan,
                loop_vehicle_count=np.nan,
                loop_occupancy=np.nan,
            ),
        ]
    )
    assert json.loads(detector.predict(dropout).iloc[0]["hints"]) == ["sensor_dropout"]
    level = RuleBasedCongestion().predict(incident).iloc[0]["level"]
    assert level in ("heavy", "standstill")


# ----------------------------------------------------- frozen baselines, level term, hold


def test_frozen_baseline_overrides_rolling_baseline() -> None:
    raw = pd.concat([rows(60), rows(20, start=T0 + pd.Timedelta(seconds=300), loop_speed_kmh=10.0)])
    raw = raw.reset_index(drop=True)
    rolling = base_features(raw, compute_references(raw))
    frozen = raw.assign(baseline_loop_speed_kmh=np.where(raw.index >= 60, 40.0, np.nan))
    held = base_features(frozen, compute_references(raw))
    # Late in the incident the rolling baseline has started absorbing it; the frozen one hasn't.
    assert held["baseline_loop_speed_kmh"].iloc[-1] == 40.0
    assert held["loop_speed_kmh_rel"].iloc[-1] == pytest.approx(0.25)
    assert (held["loop_speed_kmh_rel"].iloc[:60] == rolling["loop_speed_kmh_rel"].iloc[:60]).all()


def test_level_distances_and_excess() -> None:
    raw = pd.concat(
        [
            rows(60, stopped_vehicles=1),
            rows(5, start=T0 + pd.Timedelta(seconds=300), loop_speed_kmh=10.0, stopped_vehicles=4),
        ]
    )
    base = base_features(raw, compute_references(raw))
    dist = level_distances(base)
    assert dist.shape == (65, len(LEVEL_FEATURES))
    col = {f: i for i, f in enumerate(LEVEL_FEATURES)}
    assert dist[-1, col["loop_speed_kmh_rel"]] == pytest.approx(0.75)  # |0.25 - 1|
    assert dist[-1, col["stopped_excess"]] == pytest.approx(3.0)  # 4 over a baseline of 1
    assert dist[30, col["stopped_excess"]] == 0 and dist[30, col["loop_speed_kmh_rel"]] == 0


def test_frames_is_health_signal_not_level_or_frozen() -> None:
    # A lasting drop in processed frames (pipeline throughput) must not hold a traffic alert:
    # no level term for frames_rel, and a frozen frames baseline is ignored (it keeps rolling).
    assert "frames_rel" not in LEVEL_FEATURES
    raw = pd.concat([rows(60), rows(20, start=T0 + pd.Timedelta(seconds=300), frames=18)])
    raw = raw.reset_index(drop=True)
    frozen = raw.assign(baseline_frames=np.where(raw.index >= 60, 24.0, np.nan))
    held = base_features(frozen, compute_references(raw))
    rolling = base_features(raw, compute_references(raw))
    assert held["baseline_frames"].equals(rolling["baseline_frames"])
    assert "frames" not in current_baselines(raw)


def test_current_baselines_skip_recent_minute_and_need_history() -> None:
    raw = pd.concat(
        [
            rows(40, loop_speed_kmh=40.0),
            rows(12, start=T0 + pd.Timedelta(seconds=200), loop_speed_kmh=5.0),
        ]
    )
    b = current_baselines(raw)
    assert b["loop_speed_kmh"] == 40.0  # the last 12 windows (the unfolding incident) are excluded
    assert b["stopped_vehicles"] == 0.0
    assert current_baselines(rows(20))["loop_speed_kmh"] is None  # under BASELINE_MIN_WINDOWS


def test_level_term_scores_a_sustained_shift_the_reconstruction_misses() -> None:
    rng = np.random.default_rng(1)
    errors = rng.exponential(1.0, (2000, len(AE_FEATURES)))
    levels = np.abs(rng.normal(0, 0.05, (2000, len(LEVEL_FEATURES))))
    cams = np.array(["cam"] * 2000)
    cfg = AnomalyConfig(level_weight=1.0, threshold_percentile=99.5)
    cal = calibrate(errors, cams, cfg, levels)
    assert set(cal.level_scale) == {"__global__", "cam"}
    # A loop speed 80% below its baseline that the autoencoder reconstructs well:
    shifted = levels[:1].copy()
    shifted[0, LEVEL_FEATURES.index("loop_speed_kmh_rel")] = 0.8
    parts = score_parts(errors[:1] * 0.2, shifted, cams[:1], cal, cfg)
    assert parts.reconstruction_score[0] < cal.threshold("cam") <= parts.score[0]
    assert parts.level_score[0] == parts.score[0]
    off = score_parts(errors[:1] * 0.2, shifted, cams[:1], cal, AnomalyConfig(level_weight=0.0))
    assert off.score[0] < cal.threshold("cam")  # weight 0 switches the term off


def test_alarm_hold_keeps_alert_open() -> None:
    tracker = AlarmTracker(cooldown_s=0, clear_after=2, min_consecutive=1)
    assert tracker.update(0, 2.0, 1.0) is AlarmEvent.START
    for t in (5, 10, 15):  # score has dropped, but the hold condition still applies
        assert tracker.update(t, 0.1, 1.0, hold=True) is None
    assert tracker.active and tracker.held_checks == 3
    assert tracker.update(20, 0.1, 1.0) is None  # first clear check once released
    assert tracker.update(25, 0.1, 1.0) is AlarmEvent.END
