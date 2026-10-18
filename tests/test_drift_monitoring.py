"""Drift monitoring (ml.monitoring): per-feature drift scores on synthetic shifted and unshifted
data, the overall verdict, data quality checks, reference slicing, settings, and the
scheduler. Evidently runs for real; nothing needs Kafka, Postgres or MLflow."""

import threading
from datetime import UTC, datetime, timedelta

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("evidently")

from ml.monitoring.config import MonitorConfig  # noqa: E402
from ml.monitoring.drift import (  # noqa: E402
    CheckSpec,
    Feature,
    drift_report,
    quality_issues,
    run_check,
    seasonal_slice,
    verdict,
)
from ml.monitoring.scheduler import Scheduler  # noqa: E402

END = datetime(2026, 10, 17, 15, 0, tzinfo=UTC)
WINDOW = (END - timedelta(minutes=15), END)
EMB = tuple(f"emb_{i}" for i in range(8))
SPEC = CheckSpec(
    "test-model",
    (
        Feature("speed", "num", "data", valid_range=(0, 200), quality=True),
        Feature("count", "num", "data", valid_range=(0, 500), quality=True),
        Feature("level", "cat", "prediction"),
        Feature("text_embedding", "embedding", "data", columns=EMB),
    ),
)
CONFIG = MonitorConfig(share_threshold=0.5, min_rows=30)


def frame(n: int, seed: int, *, speed=40.0, count=10.0, levels=(0.6, 0.3, 0.1), emb=0.0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "speed": rng.normal(speed, 5.0, n).clip(0),
            "count": rng.poisson(count, n).astype(float),
            "level": rng.choice(["free_flow", "moderate", "heavy"], n, p=list(levels)),
        }
    )
    for c in EMB:
        df[c] = rng.normal(emb, 1.0, n)
    return df


def scores(results) -> dict[str, float]:
    return {r.feature: r.score for r in results}


# ----------------------------------------------------------------------------- drift scores


def test_unshifted_data_does_not_drift() -> None:
    results = drift_report(SPEC, frame(3000, 1), frame(400, 2), CONFIG)
    assert {r.feature for r in results} == {"speed", "count", "level", "text_embedding"}
    assert not [r.feature for r in results if r.drifted], scores(results)
    by = {r.feature: r for r in results}
    assert by["speed"].method == "wasserstein" and by["speed"].score < 0.1
    assert by["level"].method == "jensenshannon" and by["level"].score < 0.1
    assert by["text_embedding"].method == "model_roc_auc" and by["text_embedding"].score < 0.55


def test_shifted_data_drifts_on_the_shifted_features_only() -> None:
    current = frame(400, 2, speed=25.0, levels=(0.1, 0.3, 0.6), emb=1.0)
    results = {r.feature: r for r in drift_report(SPEC, frame(3000, 1), current, CONFIG)}
    assert results["speed"].drifted and results["speed"].score > 1.0  # ~3 reference std devs
    assert results["level"].drifted and results["level"].score > 0.3
    assert results["text_embedding"].drifted and results["text_embedding"].score > 0.8
    assert not results["count"].drifted
    assert results["speed"].current_summary["mean"] < results["speed"].reference_summary["mean"]


def test_per_feature_threshold_override() -> None:
    config = MonitorConfig(share_threshold=0.5, feature_thresholds="test-model:speed=5.0")
    current = frame(400, 2, speed=25.0)
    by = {r.feature: r for r in drift_report(SPEC, frame(3000, 1), current, config)}
    assert by["speed"].threshold == 5.0 and not by["speed"].drifted


def test_feature_defaults_method_and_threshold() -> None:
    """A feature's calibrated method and threshold apply unless DRIFT_FEATURE_THRESHOLDS
    overrides them; confidences piled up near 1 are compared as histograms."""
    spec = CheckSpec(
        "test-model",
        (Feature("confidence", "num", "prediction", method="jensenshannon", threshold=0.3),),
    )
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"confidence": 1 - rng.exponential(0.005, 2000).clip(0, 1)})
    slightly = pd.DataFrame({"confidence": 1 - rng.exponential(0.008, 300).clip(0, 1)})
    unsure = pd.DataFrame({"confidence": rng.uniform(0.4, 1.0, 300)})
    (calm,) = drift_report(spec, ref, slightly, CONFIG)
    assert calm.method == "jensenshannon" and calm.threshold == 0.3 and not calm.drifted
    (shifted,) = drift_report(spec, ref, unsure, CONFIG)
    assert shifted.drifted and shifted.score > 0.3
    strict = MonitorConfig(feature_thresholds="confidence=0.01")
    assert drift_report(spec, ref, slightly, strict)[0].threshold == 0.01
    assert strict.threshold("m", "other", "num", default=0.4) == 0.4


def test_constant_reference_feature_is_compared_as_categories() -> None:
    ref, cur = frame(1000, 1), frame(300, 2)
    ref["count"] = 0.0
    cur["count"] = np.where(np.arange(len(cur)) % 2 == 0, 0.0, 3.0)
    by = {r.feature: r for r in drift_report(SPEC, ref, cur, CONFIG)}
    assert by["count"].method == "jensenshannon" and by["count"].drifted
    assert "constant" in by["count"].note


def test_too_few_rows_for_a_feature_leave_it_unscored() -> None:
    cur = frame(300, 2)
    cur.loc[10:, "speed"] = np.nan  # 10 values left
    by = {r.feature: r for r in drift_report(SPEC, frame(1000, 1), cur, CONFIG)}
    assert by["speed"].score is None and "too few rows" in by["speed"].note


# ----------------------------------------------------------------------------- verdict


def test_run_check_ok_warning_drifted_and_html(tmp_path) -> None:
    ref = frame(3000, 1)
    ok = run_check(SPEC, "1", ref, frame(400, 2), CONFIG, WINDOW, reports_dir=tmp_path)
    assert ok.status == "ok" and ok.drift_share == 0.0
    assert ok.html_path is not None
    assert ok.html_path.read_text(encoding="utf-8").lstrip().startswith("<!DOCTYPE html>")

    one = run_check(SPEC, "1", ref, frame(400, 2, speed=25.0), CONFIG, WINDOW)
    assert one.status == "warning" and one.drift_share == 0.25
    assert [f.feature for f in one.drifted_features] == ["speed"] and one.drift_type == "data"

    many = frame(400, 2, speed=25.0, levels=(0.1, 0.3, 0.6), emb=1.0)
    drifted = run_check(SPEC, "1", ref, many, CONFIG, WINDOW)
    assert drifted.status == "drifted" and drifted.drift_share == 0.75
    record = drifted.to_dict()
    assert record["drifted_features"] == ["speed", "level", "text_embedding"]
    assert record["status"] == "drifted" and record["window_end"] == END.isoformat()


def test_run_check_without_enough_current_rows_is_no_data(tmp_path) -> None:
    result = run_check(
        SPEC, "1", frame(3000, 1), frame(10, 2), CONFIG, WINDOW, reports_dir=tmp_path
    )
    assert result.status == "no_data" and result.html_path is None
    assert all(f.score is None for f in result.features)


def test_verdict_counts_quality_issues_as_warning() -> None:
    results = drift_report(SPEC, frame(1000, 1), frame(300, 2), CONFIG)
    assert verdict(results, [], 300, CONFIG) == ("ok", 0.0)
    issue = quality_issues(SPEC, frame(10, 1), frame(10, 1).assign(speed=-1.0), CONFIG)
    assert verdict(results, issue, 300, CONFIG)[0] == "warning"


# ----------------------------------------------------------------------------- data quality


def test_quality_missing_values_against_the_reference_share() -> None:
    ref, cur = frame(1000, 1), frame(300, 2)
    ref.loc[:99, "speed"] = np.nan  # the reference itself misses 10%
    cur.loc[:44, "speed"] = np.nan  # 15%: within reference + tolerance (0.1)
    assert quality_issues(SPEC, ref, cur, CONFIG) == []
    cur.loc[:89, "speed"] = np.nan  # 30%
    (issue,) = quality_issues(SPEC, ref, cur, CONFIG)
    assert issue.kind == "missing_values" and issue.target == "speed" and issue.value == 0.3


def test_quality_out_of_range_values() -> None:
    cur = frame(300, 2)
    cur.loc[:9, "count"] = -5.0  # 3.3% > 1%
    (issue,) = quality_issues(SPEC, frame(1000, 1), cur, CONFIG)
    assert issue.kind == "out_of_range" and issue.target == "count"
    assert "outside [0, 500]" in issue.message


def test_quality_stale_cameras() -> None:
    seen = {
        "cam-a": END - timedelta(seconds=10),
        "cam-b": END - timedelta(minutes=5),
        "cam-c": None,
    }
    issues = quality_issues(SPEC, frame(100, 1), frame(100, 2), CONFIG, seen, END)
    assert [(i.kind, i.target) for i in issues] == [
        ("stale_camera", "cam-b"),
        ("stale_camera", "cam-c"),
    ]
    assert "no data in the window" in issues[1].message


# ----------------------------------------------------------------------------- reference, config


def test_seasonal_slice_matches_kind_of_day_and_time_of_day() -> None:
    # Thursday 15 and Friday 16 October (weekdays), Saturday 17 (weekend), every minute.
    ts = pd.date_range("2026-10-15", periods=3 * 24 * 60, freq="min", tz="UTC")
    ref = pd.DataFrame({"ts": ts, "speed": np.arange(len(ts), dtype=float)})
    thursday = datetime(2026, 10, 15, 15, 0, tzinfo=UTC)
    sliced, note = seasonal_slice(ref, thursday, hours=1.0, min_rows=100)
    assert set(pd.to_datetime(sliced["ts"]).dt.hour) == {14, 15, 16}
    assert set(pd.to_datetime(sliced["ts"]).dt.day) == {15, 16}  # weekdays only
    assert note.startswith("weekdays at the same time of day")
    saturday, _ = seasonal_slice(ref, END, hours=1.0, min_rows=100)  # END is a Saturday
    assert set(pd.to_datetime(saturday["ts"]).dt.day) == {17}
    # Not enough at that hour: the same kind of day at any hour.
    day, note = seasonal_slice(ref, END, hours=1.0, min_rows=1000)
    assert len(day) == 24 * 60 and note.startswith("weekends at any time of day")
    # No weekend data at all: the whole reference, and the note says why.
    weekdays = ref[pd.to_datetime(ref["ts"]).dt.day < 17]
    full, note = seasonal_slice(weekdays, END, hours=1.0, min_rows=100)
    assert len(full) == len(weekdays) and "no weekend data" in note
    assert seasonal_slice(ref, END, hours=0)[0] is ref


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRIFT_CHECK_INTERVAL_MINUTES", "5")
    monkeypatch.setenv("DRIFT_MODELS", "vehicle-detector, incident-text-classifier")
    monkeypatch.setenv("DRIFT_AUTO_RETRAIN", "true")
    monkeypatch.setenv("DRIFT_FEATURE_THRESHOLDS", "confidence=0.2,incident-text-classifier:x=0.3")
    config = MonitorConfig.from_env()
    assert config.interval_minutes == 5 and config.auto_retrain is True
    assert config.models == ("vehicle-detector", "incident-text-classifier")
    assert config.threshold("vehicle-detector", "confidence", "num") == 0.2
    assert config.threshold("incident-text-classifier", "x", "cat") == 0.3
    assert config.threshold("other", "x", "cat") == config.cat_threshold
    with pytest.raises(ValueError, match="unknown models"):
        MonitorConfig(models=("nope",))
    with pytest.raises(ValueError, match="DRIFT_SHARE_THRESHOLD"):
        MonitorConfig(share_threshold=0)


# ----------------------------------------------------------------------------- scheduler


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_scheduler_runs_at_start_then_every_interval() -> None:
    clock, runs = FakeClock(), []
    s = Scheduler(60, lambda: runs.append(clock.t), threading.Event(), clock=clock)
    assert s.run_pending() and runs == [1000.0]
    assert not s.run_pending()  # not due again yet
    clock.t = 1059.9
    assert not s.run_pending()
    clock.t = 1060.0
    assert s.run_pending() and s.next_due == 1120.0
    assert s.stats.runs == 2 and s.stats.skipped_slots == 0


def test_scheduler_skips_missed_slots_instead_of_bursting() -> None:
    clock = FakeClock()

    def slow_job() -> None:
        clock.t += 150  # overruns two and a half intervals

    s = Scheduler(60, slow_job, threading.Event(), clock=clock)
    s.run_pending()
    assert clock.t == 1150 and s.next_due == 1180  # the next slot still in the future
    assert s.stats.skipped_slots == 2
    assert not s.run_pending()


def test_scheduler_survives_failures_and_can_be_triggered() -> None:
    clock = FakeClock()

    def boom() -> None:
        raise RuntimeError("check failed")

    s = Scheduler(60, boom, threading.Event(), run_at_start=False, clock=clock)
    assert not s.run_pending()
    clock.t = 1010
    s.trigger()  # on demand
    assert s.run_pending() and s.stats.failures == 1
    assert s.next_due == 1060  # back on the regular schedule
    with pytest.raises(ValueError):
        Scheduler(0, boom, threading.Event())


def test_scheduler_run_forever_stops() -> None:
    stop, runs = threading.Event(), []

    def job() -> None:
        runs.append(1)
        stop.set()

    thread = threading.Thread(target=Scheduler(3600, job, stop).run_forever)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive() and runs == [1]


# ----------------------------------------------------------------------------- MLflow storage


def test_html_reports_go_to_mlflow_on_warning_or_drift_else_hourly() -> None:
    from ml.monitoring.monitor import Monitor

    def result(status: str, minutes: int):
        r = run_check(SPEC, "1", frame(500, 1), frame(10, 2), CONFIG, WINDOW)
        r.model_name, r.status = "test-model", status
        r.window_end = END + timedelta(minutes=minutes)
        return r

    monitor = Monitor(MonitorConfig(), None, bootstrap="unused:9092")
    monitor._last_html["test-model"] = END
    assert not monitor.html_due(result("ok", 15))
    assert monitor.html_due(result("warning", 15))
    assert monitor.html_due(result("drifted", 30))
    assert not monitor.html_due(result("no_data", 45))
    assert monitor.html_due(result("ok", 60))  # an hour since the last logged report
    monitor._last_html["test-model"] = None  # never logged
    assert monitor.html_due(result("ok", 15))
    always = Monitor(MonitorConfig(mlflow_html_every_minutes=0), None, bootstrap="unused:9092")
    always._last_html["test-model"] = END
    assert always.html_due(result("ok", 15))


def test_low_variance_feature_needs_a_practical_change_in_mean() -> None:
    spec = CheckSpec("test-model", (Feature("occupancy", "num", "data"),))
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"occupancy": rng.normal(0.20, 0.005, 3000)})  # std 2.5% of the mean
    narrower = pd.DataFrame({"occupancy": rng.normal(0.204, 0.002, 400)})  # +2%, tighter
    (small,) = drift_report(spec, ref, narrower, CONFIG)
    assert small.score > 0.5  # many reference std devs away ...
    assert not small.drifted and "low-variance" in small.note  # ... but only a 2% change
    moved = pd.DataFrame({"occupancy": rng.normal(0.22, 0.005, 400)})  # +10%
    (big,) = drift_report(spec, ref, moved, CONFIG)
    assert big.drifted and big.note is None
    off = MonitorConfig(share_threshold=0.5, low_variance_cv=0)
    assert drift_report(spec, ref, narrower, off)[0].drifted
    wide = pd.DataFrame({"occupancy": rng.normal(0.20, 0.2, 3000)})  # high variance: rule n/a
    shifted = pd.DataFrame({"occupancy": rng.normal(0.21, 0.2, 400)})
    assert drift_report(spec, wide, shifted, MonitorConfig(num_threshold=0.01))[0].note is None
