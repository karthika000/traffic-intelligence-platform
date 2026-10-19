"""Unit tests for event_simulator (no Kafka or Postgres needed)."""

import csv
import random
import sys
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "event_simulator"))

pytest.importorskip("yaml")
from event_simulator import dataset  # noqa: E402
from event_simulator.config import SimulatorConfig  # noqa: E402
from event_simulator.emitter import Emitter  # noqa: E402
from event_simulator.locations import GAZETTEERS, CameraInfo, gazetteer_for  # noqa: E402
from event_simulator.patterns import (  # noqa: E402
    RoadProfile,
    WeatherModel,
    daily_flow_fraction,
    sensor_reading,
    speed_for_flow,
)
from event_simulator.reports import (  # noqa: E402
    ACTIVE_TEMPLATES,
    CORE,
    DETAIL,
    ReportGenerator,
    Stage,
    Style,
    Variant,
)
from event_simulator.scenarios import load_scenario, load_scenarios, parse_duration  # noqa: E402
from event_simulator.simulator import Simulator  # noqa: E402
from traffic_common.enums import IncidentType, ReportSource, Severity  # noqa: E402
from traffic_common.schemas import RawEvent, SimulatorCommand, TextReport  # noqa: E402

SCENARIOS_DIR = ROOT / "services" / "event_simulator" / "scenarios"
CAMERAS = dataset.demo_cameras()
OTHER_CAMERA = CameraInfo(
    "cam-x", "Mill Road - Station Approach", "Mill Road, Cambridge", 52.2, 0.13
)


def many(n: int, variant: Variant = Variant.LIVE, seed: int = 1, **kwargs):
    gen = ReportGenerator(random.Random(seed), variant)
    rng = random.Random(seed + 1)
    return [gen.generate(rng.choice(CAMERAS), **kwargs) for _ in range(n)]


# ----------------------------------------------------------------------------- reports


def test_reports_are_varied() -> None:
    reports = many(3000)
    texts = {r.text for r in reports}
    assert len(texts) > 2900
    assert {r.incident_type for r in reports} == set(IncidentType)
    assert {r.style for r in reports} == set(Style)
    assert {r.camera_id for r in reports} == {c.id for c in CAMERAS}
    assert any("#" in r.text for r in reports if r.style is Style.SOCIAL_MEDIA)
    assert any(r.text != r.text.lower() for r in reports if r.style is Style.DRIVER)
    assert any(r.text == r.text.lower() for r in reports if r.style is Style.DRIVER)


@pytest.mark.parametrize("incident_type", [t for t in IncidentType if t is not IncidentType.NONE])
@pytest.mark.parametrize("severity", list(Severity))
def test_labels_match_text(incident_type: IncidentType, severity: Severity) -> None:
    """Without noise, the text contains the incident phrase and a severity phrase for the label."""
    gen = ReportGenerator(random.Random(3), Variant.LIVE)
    for style in Style:
        r = gen.generate(
            CAMERAS[0], incident_type=incident_type, severity=severity, style=style, noise=False
        )
        assert (r.incident_type, r.severity, r.stage) == (incident_type, severity, Stage.ACTIVE)
        assert r.core in CORE[incident_type].all()
        assert r.detail in DETAIL[incident_type][severity].all()
        assert r.core.lower() in r.text.lower()
        assert r.detail.lower() in r.text.lower()


def test_cleared_reports_are_low_severity_and_say_so() -> None:
    gen = ReportGenerator(random.Random(4))
    for style in Style:
        r = gen.generate(
            CAMERAS[1],
            incident_type=IncidentType.ACCIDENT,
            severity=Severity.CRITICAL,
            style=style,
            stage=Stage.CLEARED,
            noise=False,
        )
        assert r.severity is Severity.LOW and r.stage is Stage.CLEARED and r.detail == ""
        assert r.core.lower() in r.text.lower() or "moving again" in r.text


def test_chatter_is_none_and_low() -> None:
    for r in many(300, incident_type=IncidentType.NONE):
        assert r.incident_type is IncidentType.NONE and r.severity is Severity.LOW
        assert r.core == "" and r.detail == ""


def test_location_mention_is_exact_substring() -> None:
    reports = many(4000)
    for r in reports:
        assert r.location_mention == "" or r.location_mention in r.text, r
    assert sum(1 for r in reports if r.location_mention) / len(reports) > 0.95


def test_location_mentions_use_varied_references() -> None:
    gaz = GAZETTEERS["cam-trafalgar-square"]
    gen = ReportGenerator(random.Random(9))
    camera = next(c for c in CAMERAS if c.id == "cam-trafalgar-square")
    mentions = {
        gen.generate(camera, style=Style.DRIVER, noise=False).location_mention for _ in range(600)
    }
    for kind in (gaz.official, gaz.streets, gaz.landmarks, gaz.abbreviations, gaz.misspellings):
        assert mentions & set(kind), kind
    assert any(" near " in m for m in mentions)  # compound: street near landmark


def test_official_sources_never_misspell_places() -> None:
    misspellings = {m for g in GAZETTEERS.values() for m in g.misspellings}
    for style in (Style.OPERATOR, Style.POLICE):
        for r in many(500, style=style, seed=11):
            assert r.location_mention not in misspellings


def test_source_mapping() -> None:
    r = many(1, style=Style.DRIVER)[0]
    assert r.source is ReportSource.CITIZEN
    assert many(1, style=Style.OPERATOR)[0].source is ReportSource.OPERATOR


def test_train_variant_never_uses_holdout_phrases_and_test_does() -> None:
    holdout_core = {p for pool in CORE.values() for p in pool.holdout}
    holdout_detail = {p for sev in DETAIL.values() for pool in sev.values() for p in pool.holdout}
    holdout_places = {p for g in GAZETTEERS.values() for p in g.holdout}

    train = many(3000, Variant.TRAIN)
    assert not {r.core for r in train} & holdout_core
    assert not {r.detail for r in train} & holdout_detail
    assert not {r.location_mention for r in train} & holdout_places

    test = many(1000, Variant.TEST, seed=99)
    assert sum(r.core in holdout_core for r in test) > 150
    assert sum(r.detail in holdout_detail for r in test) > 150
    assert {r.location_mention for r in test} & holdout_places


def test_unknown_camera_gets_derived_gazetteer() -> None:
    gaz = gazetteer_for(OTHER_CAMERA)
    assert gaz.official == ["Mill Road"]
    assert "Mill Rd" in gaz.abbreviations
    r = ReportGenerator(random.Random(1)).generate(OTHER_CAMERA, noise=False)
    assert r.location_mention == "" or r.location_mention in r.text


def test_every_active_template_has_a_location() -> None:
    for style, pool in ACTIVE_TEMPLATES.items():
        for template in pool.all():
            assert "{loc}" in template or "{loc_bare}" in template, (style, template)


# ----------------------------------------------------------------------------- dataset


def test_dataset_splits(tmp_path: Path) -> None:
    train = dataset.generate_split(CAMERAS, 1500, 1, Variant.TRAIN)
    test = dataset.generate_split(CAMERAS, 400, 2, Variant.TEST, exclude={r.text for r in train})
    assert len({r.text for r in train}) == 1500
    assert not {r.text for r in train} & {r.text for r in test}
    assert {r.incident_type for r in train} == set(IncidentType)
    assert any(r.stage is Stage.CLEARED for r in train)

    path = tmp_path / "train.csv"
    dataset.write_csv(path, train)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == dataset.COLUMNS
    assert len(rows) == 1500
    for row in rows:
        assert row["incident_type"] in {t.value for t in IncidentType}
        assert row["severity"] in {s.value for s in Severity}
        assert row["source"] in {s.value for s in Style}
        assert row["location_mention"] == "" or row["location_mention"] in row["text"]


def test_dataset_is_reproducible_and_ignores_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI regenerates data/nlp from the seeds, so the files must never depend on when."""
    import event_simulator.reports as reports

    class NoClock(datetime):
        @classmethod
        def now(cls, tz=None):
            raise AssertionError("the dataset read the clock")

    monkeypatch.setattr(reports, "datetime", NoClock)
    first = dataset.generate_split(CAMERAS, 300, 5, Variant.TEST)
    again = dataset.generate_split(CAMERAS, 300, 5, Variant.TEST)
    assert [r.text for r in first] == [r.text for r in again]


def test_dataset_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert dataset.main(["--out", str(tmp_path), "--train", "300", "--test", "100"]) == 0
    out = capsys.readouterr().out
    assert "incident_type:" in out and "accident" in out
    assert (tmp_path / "reports_train.csv").exists() and (tmp_path / "reports_test.csv").exists()


# ----------------------------------------------------------------------------- traffic & weather


def at(hour: float, weekday: bool = True) -> datetime:
    day = datetime(2026, 10, 14) if weekday else datetime(2026, 10, 17)  # Wednesday / Saturday
    return day + timedelta(hours=hour)


def test_daily_pattern_has_morning_and_evening_peaks() -> None:
    flow = {h: daily_flow_fraction(at(h)) for h in (3, 8, 12.5, 17.5, 22)}
    assert flow[8] > 4 * flow[3]
    assert flow[17.5] > flow[12.5] < flow[8]
    assert flow[22] < flow[12.5]
    hours = [h / 4 for h in range(96)]
    peak_hour = max(hours, key=lambda h: daily_flow_fraction(at(h)))
    assert 16.5 <= peak_hour <= 18.5 or 7.5 <= peak_hour <= 8.5
    assert all(0 <= daily_flow_fraction(at(h)) <= 1 for h in hours)


def test_weekends_have_no_commuter_peak() -> None:
    assert daily_flow_fraction(at(8, weekday=False)) < 0.5 * daily_flow_fraction(at(8))
    assert daily_flow_fraction(at(13.5, weekday=False)) > daily_flow_fraction(at(8, weekday=False))


def test_speed_drops_at_peak_and_with_scenario_factor() -> None:
    road = RoadProfile()
    assert speed_for_flow(daily_flow_fraction(at(3)), road) > speed_for_flow(
        daily_flow_fraction(at(8)), road
    )
    rng = random.Random(0)
    normal = [sensor_reading(at(12), 15, rng, road).avg_speed_kmh for _ in range(50)]
    blocked = [
        sensor_reading(at(12), 15, rng, road, count_factor=0.3, speed_factor=0.15).avg_speed_kmh
        for _ in range(50)
    ]
    assert max(blocked) < 0.3 * min(normal)
    peak_counts = [sensor_reading(at(8), 15, rng, road).vehicle_count for _ in range(200)]
    night_counts = [sensor_reading(at(3), 15, rng, road).vehicle_count for _ in range(200)]
    assert sum(peak_counts) > 4 * sum(night_counts)


def test_weather_changes_gradually_and_follows_overrides() -> None:
    model = WeatherModel(random.Random(2), change_probability=0.05)
    states = [model.tick(at(12 + i / 240)) for i in range(1)]  # warm-up tick
    visibilities, conditions = [], set()
    for i in range(400):  # 100 minutes at a 15 s tick
        state = model.tick(at(12 + i / 240))
        visibilities.append(state.visibility_m)
        conditions.add(state.condition)
    steps = [abs(b - a) for a, b in zip(visibilities, visibilities[1:], strict=False)]
    assert len(conditions) > 1  # the weather does change...
    assert max(steps) < 1500  # ...but never jumps from clear to fog in one tick
    assert states

    model.override = {"condition": "heavy_rain", "precipitation_mm_h": 30, "visibility_m": 800}
    for i in range(60):
        state = model.tick(at(13 + i / 240))
    assert state.condition == "heavy_rain"
    assert state.precipitation_mm_h == pytest.approx(30, rel=0.05)
    assert state.visibility_m == pytest.approx(800, rel=0.2)


# ----------------------------------------------------------------------------- scenarios


def test_scenario_files_load() -> None:
    scenarios = load_scenarios(SCENARIOS_DIR)
    assert set(scenarios) == {
        "multi_vehicle_accident",
        "broken_down_lorry",
        "flooding",
        "road_works",
    }
    for s in scenarios.values():
        last = s.phases[-1]
        assert last.reports and last.reports.stage is Stage.CLEARED  # every scenario resolves
        assert len({st for p in s.phases if p.reports for st in p.reports.sources}) >= 3
    assert scenarios["flooding"].phases[0].weather is not None
    assert scenarios["multi_vehicle_accident"].peak_severity is Severity.CRITICAL


def test_timeline_is_contiguous_and_scaled() -> None:
    s = load_scenario(SCENARIOS_DIR / "multi_vehicle_accident.yaml")
    timeline = s.timeline()
    assert timeline[0].start == 0 and timeline[-1].end == pytest.approx(s.duration) == 18 * 60
    assert all(a.end == b.start for a, b in zip(timeline, timeline[1:], strict=False))
    fast = s.timeline(time_scale=60)
    assert fast[-1].end == pytest.approx(18)

    escalation = timeline[1]
    assert escalation.traffic_factors(escalation.start) == pytest.approx(
        (0.5, 0.75)
    )  # from phase 1
    assert escalation.traffic_factors(escalation.start + 30) == pytest.approx((0.325, 0.525))
    assert escalation.traffic_factors(escalation.start + 120) == pytest.approx((0.15, 0.3))
    assert timeline[0].traffic_factors(0) == pytest.approx((1.0, 1.0))  # ramps from normal


@pytest.mark.parametrize(("value", "seconds"), [(90, 90), ("90s", 90), ("4m", 240), ("1.5h", 5400)])
def test_parse_duration(value, seconds) -> None:
    assert parse_duration(value) == seconds


@pytest.mark.parametrize(
    ("yaml_text", "error"),
    [
        ("name: x\ndescription: d\nincident_type: accident\nphases: []", "at least 1"),
        (
            "name: x\ndescription: d\nincident_type: accident\nphases:\n"
            "  - {name: a, duration: soon, severity: low}",
            "invalid duration",
        ),
        (
            "name: x\ndescription: d\nincident_type: accident\nphases:\n"
            "  - {name: a, duration: 1m, severity: low,"
            " reports: {per_minute: 1, sources: {pigeon: 1}}}",
            "sources",
        ),
        (
            "name: x\ndescription: d\nincident_type: accident\nphases:\n"
            "  - {name: a, duration: 1m, severity: low, colour: red}",
            "Extra inputs",
        ),
        (
            "name: x\ndescription: d\nincident_type: accident\nphases:\n"
            "  - {name: a, duration: 1m, severity: low, weather: {condition: tornado}}",
            "weather",
        ),
        (
            "name: other\ndescription: d\nincident_type: accident\nphases:\n"
            "  - {name: a, duration: 1m, severity: low}",
            "must match the file name",
        ),
    ],
)
def test_invalid_scenarios_rejected(tmp_path: Path, yaml_text: str, error: str) -> None:
    path = tmp_path / "x.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_scenario(path)


# ----------------------------------------------------------------------------- scenario runs


class Collector:
    def __init__(self) -> None:
        self.messages: list[tuple[str, object]] = []
        self.lock = threading.Lock()

    def __call__(self, topic: str, message: object) -> None:
        with self.lock:
            self.messages.append((topic, message))


def make_simulator(sent: Collector, **config) -> Simulator:
    sim = Simulator(
        SimulatorConfig(**config),
        None,
        Emitter(sent, None, random.Random(1)),
        None,
        threading.Event(),
        random.Random(1),
    )
    for camera in CAMERAS:
        sim.add_camera(camera)
    return sim


def test_scenario_run_emits_tagged_escalating_timeline() -> None:
    sent = Collector()
    sim = make_simulator(sent)
    runner = sim.start_scenario("multi_vehicle_accident", "cam-old-street", time_scale=180)
    runner.join(timeout=20)
    assert not runner.is_alive() and runner.status.value == "completed"
    assert "cam-old-street" not in sim.runs

    reports = [m for t, m in sent.messages if isinstance(m, TextReport)]
    events = [m for t, m in sent.messages if isinstance(m, RawEvent)]
    assert len(reports) == runner.reports_emitted >= 20  # ~49 expected at these rates
    assert all(m.scenario_id == runner.run_id for _, m in sent.messages)
    assert all(m.camera_id == "cam-old-street" for _, m in sent.messages)
    assert {r.source for r in reports} >= {
        ReportSource.POLICE,
        ReportSource.OPERATOR,
        ReportSource.SOCIAL_MEDIA,
    }

    sensors = [e for e in events if e.event_type == "loop_detector"]
    assert {e.event_type for e in events} == {"weather", "loop_detector"}
    assert len(sensors) >= 30  # one per (15 s / time_scale)
    timeline = runner.timeline

    def speeds(phase_index: int) -> list[float]:
        # Readings carry no phase, so bucket by emission order across the scaled timeline.
        n = len(sensors)
        lo, hi = (
            int(n * w / timeline[-1].end)
            for w in (timeline[phase_index].start, timeline[phase_index].end)
        )
        return [s.payload["avg_speed_kmh"] for s in sensors[lo:hi]]

    critical = speeds(2)  # emergency_response: speed_factor 0.2
    resolved = speeds(4)[len(speeds(4)) // 2 :]  # second half of resolution, factor ~0.95
    assert sum(critical) / len(critical) < 0.5 * sum(resolved) / len(resolved)


def test_every_run_ends_with_a_clearing_report() -> None:
    """The first report of a cleared-stage phase comes as the phase starts, so even a run sped
    up so far that its final phase lasts a fraction of a second says it's over."""
    sent = Collector()
    sim = make_simulator(sent)
    stages: list[str] = []
    report = sim.emitter.report

    def spy(camera, generated, scenario_id=None, phase=None):
        stages.append(generated.stage.value)
        return report(camera, generated, scenario_id=scenario_id, phase=phase)

    sim.emitter.report = spy
    runner = sim.start_scenario("flooding", "cam-hyde-park-corner", time_scale=3000)
    runner.join(timeout=20)
    assert runner.status.value == "completed"
    assert "cleared" in stages


def test_flooding_overrides_weather_then_releases_it() -> None:
    sent = Collector()
    sim = make_simulator(sent)
    runner = sim.start_scenario("flooding", "cam-hyde-park-corner", time_scale=300)
    runner.join(timeout=20)
    weather = [
        m.payload for _, m in sent.messages if isinstance(m, RawEvent) and m.event_type == "weather"
    ]
    assert any(w["condition"] == "heavy_rain" for w in weather)
    assert max(w["precipitation_mm_h"] for w in weather) > 15
    assert sim.cameras["cam-hyde-park-corner"].weather.override is None


def test_one_scenario_per_camera_and_cancel() -> None:
    sent = Collector()
    sim = make_simulator(sent)
    runner = sim.start_scenario("road_works", "cam-elephant-castle", time_scale=1)
    with pytest.raises(ValueError, match="already runs"):
        sim.start_scenario("flooding", "cam-elephant-castle")
    with pytest.raises(ValueError, match="unknown scenario"):
        sim.start_scenario("alien_landing", "cam-old-street")
    with pytest.raises(ValueError, match="unknown camera"):
        sim.start_scenario("flooding", "cam-nowhere")
    assert sim.stop_scenario(runner.run_id)
    runner.join(timeout=10)
    assert runner.status.value == "cancelled"
    assert not sim.stop_scenario(uuid.uuid4())


def test_control_commands(caplog: pytest.LogCaptureFixture) -> None:
    sent = Collector()
    sim = make_simulator(sent)
    stale = SimulatorCommand(
        action="start_scenario",
        scenario="flooding",
        camera_id="cam-old-street",
        timestamp=datetime.now(UTC) - timedelta(hours=1),
    )
    sim.handle_command(stale)
    assert not sim.runs and "stale" in caplog.text

    sim.handle_command(
        SimulatorCommand(action="start_scenario", scenario="flooding", camera_id="cam-unknown")
    )
    assert not sim.runs and "unknown camera" in caplog.text

    cmd = SimulatorCommand(
        action="start_scenario", scenario="road_works", camera_id="cam-old-street", time_scale=2
    )
    sim.handle_command(cmd)
    assert sim.runs["cam-old-street"].run_id == cmd.scenario_id
    sim.handle_command(SimulatorCommand(action="stop_scenario", scenario_id=cmd.scenario_id))
    sim.runs.get("cam-old-street") and sim.runs["cam-old-street"].join(timeout=10)
    assert "cam-old-street" not in sim.runs


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPORT_INTERVAL_SECONDS", "2.5")
    monkeypatch.setenv("SCENARIO_SCHEDULE_MINUTES", "30")
    monkeypatch.setenv("DATASET_DIR", "/tmp/nlp")
    cfg = SimulatorConfig.from_env()
    assert (cfg.report_interval_seconds, cfg.scenario_schedule_minutes) == (2.5, 30.0)
    assert cfg.dataset_dir == Path("/tmp/nlp")
    assert cfg.sensor_interval_seconds == 15.0
    with pytest.raises(ValueError):
        SimulatorConfig(scenario_schedule_minutes=-1)
