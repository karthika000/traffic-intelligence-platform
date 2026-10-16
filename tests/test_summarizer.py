"""Incident summaries: context building (scratch database), the template, the faithfulness
guard (invented numbers, places, vehicles, contradictions must fail), the fallback paths, and
the service's debounce. No model download: generation is faked."""

import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "summarizer")]

pytest.importorskip("rapidfuzz")
pytest.importorskip("mlflow")

from ml.summarize import faithfulness  # noqa: E402
from ml.summarize.context import IncidentContext, load_context  # noqa: E402
from ml.summarize.generator import GenerationTimeout  # noqa: E402
from ml.summarize.prompts import PromptTemplate  # noqa: E402
from ml.summarize.summarizer import IncidentSummarizer, SummarizerConfig, tidy  # noqa: E402
from ml.summarize.template import template_headline, template_summary  # noqa: E402

T0 = datetime(2026, 10, 16, 17, 48, tzinfo=UTC)  # 18:48 in London


def context(**overrides) -> IncidentContext:
    values = dict(
        incident_id=str(uuid.uuid4()),
        version=2,
        camera_id="cam-hyde-park-corner",
        location="Hyde Park Corner - Piccadilly approach",
        area="Hyde Park Corner, London W1J",
        incident_type="accident",
        severity="critical",
        status="confirmed",
        start_time=T0,
        reference_time=T0 + timedelta(minutes=6),
        congestion="heavy",
        speed_kmh=6.0,
        baseline_speed_kmh=24.0,
        loop_speed_kmh=5.0,
        loop_baseline_kmh=30.0,
        stopped_vehicles=3,
        pedestrians_in_roadway=1,
        n_reports=9,
        sources=["citizen", "police", "social_media"],
        vehicles=["bus", "taxi"],
        lanes=["two lanes"],
        places=["Piccadilly"],
        fused_confidence=0.86,
        vision_score=0.74,
        text_score=0.92,
        agreement_score=0.8,
        vision_present=True,
        vision_degraded=False,
    )
    values.update(overrides)
    return IncidentContext(**values)


# ----------------------------------------------------------------------------- context


def test_facts_text_lists_only_known_facts() -> None:
    facts = context().facts_text()
    assert (
        "Location: Hyde Park Corner - Piccadilly approach (Hyde Park Corner, London W1J)" in facts
    )
    assert "Incident: accident, severity critical, status confirmed" in facts
    assert "Started: 18:48, 6 minutes ago" in facts  # London time
    assert "loop detector speed 5 km/h against 30 km/h normally (17 percent)" in facts
    assert "camera speed 6 km/h against 24 km/h normally (25 percent)" in facts  # agrees
    assert "3 stopped vehicles" in facts and "1 pedestrians in the roadway" in facts
    assert "Reports: 9 from 3 sources (drivers, police, social media)" in facts
    assert "Vehicles mentioned: bus, taxi" in facts
    assert "Recommended action: dispatch emergency response and divert traffic" in facts
    bare = context(
        congestion=None,
        speed_kmh=None,
        loop_speed_kmh=None,
        stopped_vehicles=None,
        n_reports=0,
        sources=[],
        vehicles=[],
        lanes=[],
        places=[],
        vision_present=False,
        pedestrians_in_roadway=None,
    ).facts_text()
    assert "Traffic:" not in bare and "Reports:" not in bare and "Evidence:" not in bare


@pytest.mark.parametrize(
    ("status", "severity", "action"),
    [
        ("resolved", "critical", "monitor for residual queues; no further action needed"),
        ("suspected", "critical", "verify on the camera feed before acting"),
        ("confirmed", "critical", "dispatch emergency response and divert traffic"),
        ("confirmed", "high", "divert traffic and dispatch a patrol to assess"),
        ("confirmed", "medium", "monitor and prepare a diversion"),
        ("confirmed", "low", "monitor"),
    ],
)
def test_recommended_action(status, severity, action) -> None:
    assert context(status=status, severity=severity).recommended_action == action


def test_loop_speed_preferred_when_camera_disagrees() -> None:
    # The looped footage doesn't show the incident: camera at its usual speed, loops at 17%.
    ctx = context(speed_kmh=8.0, baseline_speed_kmh=6.0)
    assert not ctx.camera_agrees and not ctx.show_camera_speed
    facts, summary = ctx.facts_text(), template_summary(ctx)
    assert "loop detector speed 5 km/h against 30 km/h normally" in facts
    assert "camera speed" not in facts and "8 km/h" not in facts
    assert "loop detectors measure 5 km/h against 30 km/h normally" in summary
    assert "camera speeds" not in summary
    # No loop data: the camera is all there is.
    no_loop = context(loop_speed_kmh=None, loop_baseline_kmh=None)
    assert no_loop.show_camera_speed and "camera speed 6 km/h" in no_loop.facts_text()
    assert "camera speeds are 6 km/h against 24 km/h normally" in template_summary(no_loop)
    # Agreeing camera: both are stated.
    assert context().show_camera_speed and "camera speeds are 6 km/h" in template_summary(context())


def test_context_round_trips() -> None:
    ctx = context()
    assert IncidentContext.from_dict(ctx.to_dict()) == ctx


def test_load_context_from_postgres(scratch_database: str) -> None:
    from alembic import command
    from sqlalchemy import text

    from traffic_common.db import create_db_engine
    from traffic_common.db.migrate import alembic_config
    from traffic_common.db.seed import seed_cameras

    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    seed_cameras(engine)
    cam, incident_id = "cam-old-street", uuid.uuid4()
    report_ids = [uuid.uuid4() for _ in range(3)]
    state = {
        "baseline": {"avg_speed_kmh": 20.0, "stopped_vehicles": 0},
        "reports": {
            str(r): {"report_id": str(r), "source": s, "stage": "active"}
            for r, s in zip(report_ids, ["citizen", "operator", "citizen"], strict=True)
        },
    }
    components = {
        "vision": {"score": 0.7, "present": True},
        "text": {"score": 0.8},
        "agreement": {"score": 0.8},
    }
    import json

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incidents (id, camera_id, status, incident_type, version, severity, "
                "fused_confidence, confidence_components, start_time, suspected_at, title, "
                "details) VALUES (:id, :c, 'confirmed', 'road_works', 3, 'high', 0.81, "
                "CAST(:comp AS jsonb), :t, :t, 'x', CAST(:d AS jsonb))"
            ),
            {
                "id": incident_id,
                "c": cam,
                "comp": json.dumps(components),
                "t": T0,
                "d": json.dumps({"state": state}),
            },
        )
        for r, ents in zip(
            report_ids,
            [
                [{"text": "van", "label": "VEHICLE", "start": 0, "end": 3}],
                [
                    {"text": "lane one", "label": "LANE", "start": 0, "end": 8},
                    {"text": "City Road", "label": "LOCATION", "start": 9, "end": 18},
                ],
                [{"text": "vans", "label": "VEHICLE", "start": 0, "end": 4}],
            ],
            strict=True,
        ):
            conn.execute(
                text(
                    "INSERT INTO reports (id, source, text, reported_at) VALUES "
                    "(:id, 'citizen', 'x', :t)"
                ),
                {"id": r, "t": T0},
            )
            conn.execute(
                text(
                    "INSERT INTO nlp_results (report_id, incident_type, model_name, "
                    "model_version, processed_at, entities) VALUES (:r, 'road_works', 'm', '1', "
                    ":t, CAST(:e AS jsonb))"
                ),
                {"r": r, "t": T0, "e": json.dumps(ents)},
            )
        # Loop detectors: 40 km/h in the 10 minutes before the start, 8 km/h now.
        for minutes, speed in [(-9, 40.0), (-6, 40.0), (-3, 40.0), (4.5, 8.0), (4.75, 8.0)]:
            conn.execute(
                text(
                    "INSERT INTO sensor_readings (event_id, camera_id, reading_at, avg_speed_kmh) "
                    "VALUES (:e, :c, :t, :v)"
                ),
                {"e": uuid.uuid4(), "c": cam, "t": T0 + timedelta(minutes=minutes), "v": speed},
            )
        for i in range(6):
            end = T0 + timedelta(minutes=5) - timedelta(seconds=5 * i)
            conn.execute(
                text(
                    "INSERT INTO traffic_metrics (camera_id, window_start, window_end, "
                    "vehicle_count, avg_speed_kmh, stopped_vehicles, pedestrians_in_roadway) "
                    "VALUES (:c, :s, :e, 5, :v, :st, 0)"
                ),
                {"c": cam, "s": end - timedelta(seconds=5), "e": end, "v": 10.0 + i, "st": i % 3},
            )
    with engine.connect() as conn:
        ctx = load_context(
            conn, incident_id, overrides={"timestamp": T0 + timedelta(minutes=5), "version": 3}
        )
    engine.dispose()
    assert ctx.location == "Old Street Roundabout - City Road" and ctx.version == 3
    assert (ctx.incident_type, ctx.severity, ctx.status) == ("road_works", "high", "confirmed")
    assert ctx.duration_min == 5 and ctx.speed_kmh == 12.5 and ctx.baseline_speed_kmh == 20.0
    assert (ctx.loop_speed_kmh, ctx.loop_baseline_kmh, ctx.loop_pct) == (8.0, 40.0, 20)
    assert not ctx.show_camera_speed  # camera at 62% of normal, loops at 20%
    assert ctx.stopped_vehicles == 2 and ctx.n_reports == 3
    assert ctx.sources == ["citizen", "operator"]
    assert ctx.vehicles == ["van"] and ctx.lanes == ["lane one"] and ctx.places == []
    assert ctx.vision_present and ctx.fused_confidence == pytest.approx(0.81)


# ----------------------------------------------------------------------------- template


def test_template_is_short_faithful_and_complete() -> None:
    for status in ("suspected", "confirmed", "resolved"):
        for severity in ("low", "medium", "high", "critical"):
            ctx = context(status=status, severity=severity)
            headline, summary = template_headline(ctx), template_summary(ctx)
            assert len(headline.split()) < 15
            n = summary.count(". ") + 1
            assert 2 <= n <= 4
            assert "Recommended action:" in summary
            verdict = faithfulness.check(f"{headline}\n{summary}", ctx)
            assert verdict.passed, (status, severity, verdict.violations)


# ----------------------------------------------------------------------------- faithfulness


@pytest.mark.parametrize(
    "output",
    [
        "Accident at Hyde Park Corner, severity critical, 3 stopped vehicles, speeds 6 km/h.",
        "A critical accident at Hyde Park Corner near Piccadilly; police and drivers report it.",
        "Crash at Hyde Park Corner since 18:48, traffic heavy. Recommended action: divert.",
    ],
)
def test_faithful_outputs_pass(output) -> None:
    assert faithfulness.check(output, context()).passed


@pytest.mark.parametrize(
    ("output", "violation"),
    [
        ("Accident at Hyde Park Corner with 7 stopped vehicles.", "number 7"),
        ("Accident since 18:30 at Hyde Park Corner.", "number 18:30"),
        ("Accident at Hyde Park Corner, about twenty reports.", "twenty"),
        ("Accident at Hyde Park Corner, started at 6pm.", "am/pm"),
        ("Accident on Park Lane near Hyde Park Corner.", "park lane"),
        ("Accident at Hyde Park Corner, queues back to Baker Street.", "Baker Street"),
        ("Accident at Hyde Park Corner involving a lorry.", "lorry"),
        ("Flooding at Hyde Park Corner.", "weather hazard"),
        ("Minor accident at Hyde Park Corner.", "severity 'low'"),
        ("The accident at Hyde Park Corner has been cleared.", "says it is over"),
        ("A cyclist and a pedestrian are injured in a crash at Hyde Park Corner.", "injured"),
        ("Accident at Hyde Park Corner, two people killed.", "killed"),
    ],
)
def test_invented_or_contradicting_outputs_fail(output, violation) -> None:
    verdict = faithfulness.check(output, context())
    assert not verdict.passed
    assert any(violation.lower() in v.lower() for v in verdict.violations), verdict.violations


def test_resolved_incident_must_not_be_called_ongoing() -> None:
    v = faithfulness.check(
        "The accident is ongoing at Hyde Park Corner.", context(status="resolved")
    )
    assert not v.passed


# ----------------------------------------------------------------------------- fallback paths


class FakeGenerator:
    revision = "fake-1"

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result, self.error, self.calls = result, error, 0

    def generate(self, ctx, timeout_s):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def summarizer(generator) -> IncidentSummarizer:
    return IncidentSummarizer(SummarizerConfig(model_name="fake/model"), generator)


def test_faithful_model_output_is_used() -> None:
    gen = FakeGenerator(
        (
            "Critical accident at Hyde Park Corner",
            "An accident at Hyde Park Corner is blocking traffic. "
            "Congestion is heavy with 3 stopped vehicles.",
        )
    )
    r = summarizer(gen).summarize(context())
    assert r.generation == "model" and r.model_version == "fake-1"
    assert r.prompt_version == PromptTemplate.load("v1").version
    assert r.summary.endswith("Recommended action: dispatch emergency response and divert traffic.")
    assert (
        r.faithfulness["passed"] and "recommended action appended" in r.faithfulness["adjustments"]
    )


@pytest.mark.parametrize(
    ("generator", "reason"),
    [
        (
            FakeGenerator(
                (
                    "Accident at Hyde Park Corner",
                    "A crash with 12 cars on Park Lane. Traffic is heavy.",
                )
            ),
            "faithfulness",
        ),
        (FakeGenerator(error=GenerationTimeout("took too long")), "timeout"),
        (FakeGenerator(error=RuntimeError("out of memory")), "error: RuntimeError"),
        (FakeGenerator(("Accident", "Accident.")), "too short"),
        (
            FakeGenerator(
                (
                    "Traffic congestion heavy; camera speed 6 km/h",
                    "Traffic at Hyde Park Corner is heavy with 3 stopped vehicles.",
                )
            ),
            "headline doesn't name the incident type",
        ),
        (
            FakeGenerator(
                ("Critical accident reported", "An accident is blocking traffic, which is heavy.")
            ),
            "location not named",
        ),
        (
            FakeGenerator(
                (
                    " ".join(["Critical accident at Hyde Park Corner"] * 3),
                    "An accident at Hyde Park Corner is blocking traffic, which is heavy.",
                )
            ),
            "headline too long",
        ),
    ],
)
def test_fallback_to_template(generator, reason) -> None:
    ctx = context()
    r = summarizer(generator).summarize(ctx)
    assert r.generation == "template" and r.model_name == "incident-summary-template"
    assert reason in r.faithfulness["fallback_reason"]
    assert r.headline == template_headline(ctx) and r.summary == template_summary(ctx)
    if reason == "faithfulness":
        assert any("park lane" in v for v in r.faithfulness["model_violations"])
        assert r.faithfulness["model_output"]["summary"].startswith("A crash")


def test_template_configuration_never_calls_a_model() -> None:
    s = IncidentSummarizer(SummarizerConfig(generator="template"))
    r = s.summarize(context())
    assert r.generation == "template" and s.generator is None
    assert r.faithfulness["fallback_reason"] == "template configuration"


def test_tidy_enforces_shape() -> None:
    long = " ".join(["word"] * 20)
    headline, summary, notes = tidy(
        long, "One. Two. Three. Four. Five. Recommended action: x.", context()
    )
    assert len(headline.split()) == 14 and summary.count(".") == 4
    assert "headline too long" in notes and "trimmed to 4 sentences" in notes


def test_chat_prompt_and_output_parsing() -> None:
    from ml.summarize.generator import parse_chat_output

    p = PromptTemplate.load("v2")
    assert p.layout == "chat" and PromptTemplate.load("v1").layout == "seq2seq"
    messages = p.messages("FACTS")
    assert [m["role"] for m in messages] == ["system", "user"] and "FACTS" in messages[1]["content"]
    assert parse_chat_output("Headline: Crash at Old Street\nSummary: A crash.\n It is slow.") == (
        "Crash at Old Street",
        "A crash. It is slow.",
    )
    assert parse_chat_output("**Headline:** Crash\n**Summary:** Slow traffic.") == (
        "Crash",
        "Slow traffic.",
    )
    assert parse_chat_output("Crash at Old Street\nTraffic is slow.") == (
        "Crash at Old Street",
        "Traffic is slow.",
    )


def test_prompt_versions_are_content_addressed() -> None:
    p = PromptTemplate.load("v1")
    assert p.version.startswith("v1-") and len(p.version) == 11
    assert "{facts}" not in p.render("summary", "FACTS") and "FACTS" in p.render("summary", "FACTS")


# ----------------------------------------------------------------------------- debounce


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def incident(version: int, status="suspected", t="accident", severity="high", iid=None) -> dict:
    return {
        "incident_id": iid or uuid.UUID(int=1),
        "version": version,
        "status": status,
        "incident_type": t,
        "severity": severity,
    }


def test_debounce_summarises_only_the_latest_version() -> None:
    from summarizer_service.config import SummarizerServiceConfig
    from summarizer_service.service import SummarizerService

    clock = Clock()
    svc = SummarizerService(
        SummarizerServiceConfig(debounce_seconds=10), None, None, None, monotonic=clock
    )
    svc.offer(incident(1))
    clock.t = 4
    svc.offer(incident(2, status="confirmed"))  # within 10 s: replaces v1, restarts the timer
    clock.t = 12
    assert svc.due() == []  # 8 s after v2
    clock.t = 14.5
    [p] = svc.due()
    assert p.incident["version"] == 2
    assert p.reasons == {"created", "status suspected -> confirmed"}
    svc.last[str(p.incident["incident_id"])] = ("confirmed", "accident", "high")
    svc.offer(incident(3, status="confirmed"))  # confidence-only update: ignored
    assert svc.pending == {}
    svc.offer(incident(4, status="confirmed", severity="critical"))
    assert set(svc.pending) == {str(uuid.UUID(int=1))}
    other = uuid.UUID(int=2)
    svc.offer(incident(1, iid=other))  # incidents are debounced independently
    assert len(svc.pending) == 2
