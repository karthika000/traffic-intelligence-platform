"""NLP building blocks without PyTorch or model downloads: preprocessing, label maps, entity
rules, camera matching, the keyword fallback, and nlp_service batching (scratch database)."""

import json
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "services" / "nlp_service")]
# Generated (data/ is git-ignored): python -m event_simulator.dataset, as CI does first.
REPORTS_TEST = ROOT / "data" / "nlp" / "reports_test.csv"

pytest.importorskip("rapidfuzz")
pd = pytest.importorskip("pandas")

from ml.nlp.entities import LANE, LOCATION, TIME, VEHICLE, location_spans, rule_entities  # noqa: E402
from ml.nlp.labels import label_maps, preprocess  # noqa: E402
from ml.nlp.matching import CameraMatcher, normalize  # noqa: E402
from ml.nlp.predict import OUTPUT_COLUMNS  # noqa: E402
from ml.nlp.rules import KeywordIncidentClassifier  # noqa: E402
from traffic_common.db.seed import DEMO_CAMERAS  # noqa: E402
from traffic_common.enums import IncidentType, Severity  # noqa: E402
from traffic_common.gazetteer import GAZETTEERS, CameraInfo  # noqa: E402

CAMERAS = [
    CameraInfo(c["id"], c["name"], c["location_name"], c["latitude"], c["longitude"])
    for c in DEMO_CAMERAS
]


@pytest.fixture(scope="module")
def matcher() -> CameraMatcher:
    return CameraMatcher(CAMERAS)


# ----------------------------------------------------------------------------- preprocessing


def test_preprocess_splits_hashtags_and_normalises() -> None:
    text = "Crash  on  #HydeParkCorner\n see https://t.co/abc ＃ok #londontraffic"
    assert preprocess(text) == "Crash on Hyde Park Corner see url ok londontraffic"
    assert preprocess("  ") == ""


def test_label_maps_follow_the_enums_and_round_trip() -> None:
    maps = label_maps()
    assert maps["incident_type"].labels == tuple(t.value for t in IncidentType)
    assert maps["severity"].labels == ("low", "medium", "high", "critical")
    ids = maps["severity"].encode(["high", "low", "critical"])
    assert ids.tolist() == [2, 0, 3]
    assert maps["severity"].decode(ids) == ["high", "low", "critical"]
    with pytest.raises(ValueError, match="unknown labels"):
        maps["incident_type"].encode(["alien_landing"])
    restored = type(maps["incident_type"]).from_dict(maps["incident_type"].to_dict())
    assert restored == maps["incident_type"] and len(restored) == 8


# ----------------------------------------------------------------------------- entities


def test_rule_entities_find_vehicles_lanes_and_times() -> None:
    text = (
        "Articulated lorry and a black cab collided at 7:45am, lane two and the bus lane "
        "blocked northbound, delays for 20 mins"
    )
    found = {(e["label"], e["text"]) for e in rule_entities(text)}
    assert {
        (VEHICLE, "Articulated lorry"),
        (VEHICLE, "black cab"),
        (TIME, "7:45am"),
        (LANE, "lane two"),
        (LANE, "bus lane"),
        (LANE, "northbound"),
        (TIME, "for 20 mins"),
    } <= found
    for e in rule_entities(text):
        assert text[e["start"] : e["end"]] == e["text"]
    assert rule_entities("nothing to see here") == []


def test_rule_entities_prefer_the_longest_overlapping_span() -> None:
    labels = [(e["label"], e["text"]) for e in rule_entities("two lanes closed by a tow truck")]
    assert labels == [(LANE, "two lanes closed"), (VEHICLE, "tow truck")]


def test_location_spans_keep_loc_and_confirmed_org() -> None:
    text = "Crash at Wellington Arch, Met Police attending"
    ner = [
        {"text": "Wellington Arch", "label": "ORG", "start": 9, "end": 24},
        {"text": "Met Police", "label": "ORG", "start": 26, "end": 36},
    ]
    gaz = [{"text": "Wellington Arch", "label": LOCATION, "start": 9, "end": 24}]
    spans = location_spans(text, ner, gaz)
    assert [(s["text"], s["label"]) for s in spans] == [("Wellington Arch", LOCATION)]
    loc = [{"text": "Brook Drive", "label": "LOC", "start": 0, "end": 11}]
    assert [s["text"] for s in location_spans("Brook Drive closed", loc, [])] == ["Brook Drive"]


# ----------------------------------------------------------------------------- matching


def test_normalize_expands_abbreviations() -> None:
    assert normalize("Old St rbt") == "old street roundabout"
    assert normalize("E&C") == "e and c"
    assert normalize("Hyde Pk Cnr") == "hyde park corner"
    assert normalize("St George's Road") == "street georges road"
    assert normalize("#ElephantAndCastle") == "elephant and castle"


@pytest.mark.parametrize(
    ("text", "camera"),
    [
        ("Crash at Trafalger Sqaure, two lanes shut", "cam-trafalgar-square"),  # two typos
        ("queues back to Hyde Prk Corner", "cam-hyde-park-corner"),
        ("broken down van on picadilly", "cam-hyde-park-corner"),  # misspelt street
        ("debris approaching Old St rbt", "cam-old-street"),  # abbreviations
        ("stuck near E&C again", "cam-elephant-castle"),  # short alias, exact
        ("flooding nr Elephnt & Castle", "cam-elephant-castle"),  # unseen typo
        ("total gridlock 🚗🚗 #OldStreet #LondonTraffic", "cam-old-street"),  # hashtag only
        ("lorry shed its load on Newington Cswy", "cam-elephant-castle"),
        ("POLICE INCIDENT AT CHARING X", "cam-trafalgar-square"),
    ],
)
def test_matcher_tolerates_typos_and_abbreviations(matcher, text, camera) -> None:
    result = matcher.match(text)
    assert result.camera_id == camera, result
    assert 0.7 <= result.confidence <= 1.0
    assert text[result.start : result.end] == result.matched_text


def test_matcher_returns_none_without_a_location(matcher) -> None:
    for text in ("running late again, nothing to do with traffic", "Crash on Brook Drive"):
        result = matcher.match(text)
        assert result.camera_id is None
    assert matcher.gazetteer_entities("nothing here") == []


def test_matcher_ignores_holdout_forms(matcher) -> None:
    holdout = {matcher_alias.norm for matcher_alias in matcher.aliases}
    for gaz in GAZETTEERS.values():
        for form in gaz.holdout:
            assert normalize(form) not in holdout  # the test split's unseen forms stay unseen


def test_ambiguous_aliases_and_close_calls_lower_confidence() -> None:
    a = CameraInfo("cam-a", "Bridge Street - North", "Bridge Street, Town", 51.5, -0.1)
    b = CameraInfo("cam-b", "Bridge Street - South", "Bridge Street, Town", 51.4, -0.1)
    m = CameraMatcher([a, b])
    assert "bridge street" not in {x.norm for x in m.aliases}  # shared: identifies neither
    assert m.match("crash on Bridge Street").camera_id is None
    north = m.match("crash on Bridge Street North")
    assert north.camera_id == "cam-a"  # derived from the camera name, unique


def test_derived_gazetteer_cameras_can_be_matched() -> None:
    cam = CameraInfo("cam-x", "Kings Cross - York Way", "Kings Cross, London N1", 51.53, -0.12)
    result = CameraMatcher([*CAMERAS, cam]).match("van broken down on York Way")
    assert result.camera_id == "cam-x"


def test_camera_matching_accuracy_on_the_test_split(matcher) -> None:
    """Guards against regressions on the held-out set (unseen location forms are ~30%)."""
    if not REPORTS_TEST.exists():
        pytest.skip(
            f"{REPORTS_TEST.relative_to(ROOT).as_posix()} missing; generate it with "
            "`python -m event_simulator.dataset` (in services/event_simulator)"
        )
    test = pd.read_csv(REPORTS_TEST)
    predicted = [matcher.match(t).camera_id for t in test.text]
    correct = sum(p == c for p, c in zip(predicted, test.camera_id, strict=True))
    wrong = sum(p is not None and p != c for p, c in zip(predicted, test.camera_id, strict=True))
    assert correct / len(test) >= 0.85
    assert wrong / len(test) <= 0.01  # when it names a camera, it is almost always right


# ----------------------------------------------------------------------------- fallback


def test_keyword_fallback_outputs_the_shared_format() -> None:
    out = KeywordIncidentClassifier().predict(
        pd.DataFrame(
            {"text": ["Serious crash, all lanes closed", "lovely sunny day", "long queues"]}
        )
    )
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.incident_type.tolist() == ["accident", "none", "congestion"]
    assert out.severity[0] == "critical"
    probs = json.loads(out.probabilities[0])
    assert set(probs) == {"incident_type", "severity"}
    assert abs(sum(probs["incident_type"].values()) - 1) < 1e-3
    assert set(out.severity) <= {s.value for s in Severity}


# ----------------------------------------------------------------------------- service


class CountingClassifier(KeywordIncidentClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []

    def predict(self, model_input):
        self.calls.append(len(model_input))
        return super().predict(model_input)


@pytest.fixture
def nlp(scratch_database: str):
    from alembic import command

    from ml.registry import ModelManager
    from nlp_service.config import NlpConfig
    from nlp_service.service import NlpService
    from traffic_common.db import create_db_engine
    from traffic_common.db.migrate import alembic_config
    from traffic_common.db.seed import seed_cameras

    command.upgrade(alembic_config(scratch_database), "head")
    engine = create_db_engine(scratch_database)
    seed_cameras(engine)
    classifier = CountingClassifier()

    class NoRegistry:
        def get_registered_model(self, name):
            from mlflow.exceptions import MlflowException
            from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

            raise MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)

    models = ModelManager(
        "http://unused",
        {"classifier": "incident-text-classifier"},
        client=NoRegistry(),
        fallbacks={"classifier": lambda: classifier},
    )
    models.refresh()
    sent: list = []
    svc = NlpService(NlpConfig(ner_enabled=False), engine, models, lambda t, m: sent.append(m))
    yield svc, classifier, sent, engine
    engine.dispose()


def reports(*texts: str, camera: str | None = None):
    from traffic_common.schemas import TextReport

    return [TextReport(source="citizen", text=t, camera_id=camera) for t in texts]


def test_service_classifies_a_batch_in_one_call(nlp) -> None:
    from sqlalchemy import text

    svc, classifier, sent, engine = nlp
    batch = reports(
        "Crash on Trafalgr Sq, two lanes blocked by a lorry",
        "#HydeParkCorner gridlock since 8am",
        "lovely weather today",
        camera="cam-old-street",  # the report's own tag is not used for matching
    )
    svc.handle_batch(batch)
    assert classifier.calls == [3]  # one classifier call for the whole batch
    assert [m.report_id for m in sent] == [r.report_id for r in batch]
    assert [m.camera_id for m in sent] == ["cam-trafalgar-square", "cam-hyde-park-corner", None]
    first = sent[0]
    assert first.incident_type == "accident" and first.model_name == "incident-text-rules"
    assert {(e.label, e.text) for e in first.entities} >= {
        ("LOCATION", "Trafalgr Sq"),
        ("LANE", "two lanes blocked"),
        ("VEHICLE", "lorry"),
    }
    assert first.details["reported_camera_id"] == "cam-old-street"
    assert first.details["batch_size"] == 3 and first.camera_confidence >= 0.7
    with engine.connect() as conn:
        assert conn.scalar(text("SELECT count(*) FROM reports")) == 3
        rows = conn.execute(
            text("SELECT camera_id, incident_type, entities FROM nlp_results ORDER BY id")
        ).all()
    assert [r.camera_id for r in rows] == ["cam-trafalgar-square", "cam-hyde-park-corner", None]
    assert rows[0].incident_type == "accident" and rows[0].entities

    svc.handle_batch(batch[:2])  # redelivery: already analysed by this model version
    assert classifier.calls == [3] and len(sent) == 3  # nothing left to classify
    assert svc.stats.skipped_duplicates == 2


def test_service_handles_unknown_reported_cameras(nlp) -> None:
    svc, _, sent, _ = nlp
    svc.handle(reports("crash on Old Steet", camera=f"cam-{uuid.uuid4().hex[:6]}")[0])
    assert sent[0].camera_id == "cam-old-street"  # stored with camera_id NULL in reports


def test_ner_spans_snap_to_whole_words() -> None:
    from ml.nlp.ner import snap_to_words

    text = "traffic near the Faraday Memorial AGAIN, van GONE"
    i = text.index("Faraday")
    assert snap_to_words(text, i, text.index("AGAIN") + 2) == (i, i + len("Faraday Memorial"))
    j = text.index("GONE")
    assert snap_to_words(text, j, j + 2) is None  # only a fragment
    assert snap_to_words(text, i, i + len("Faraday Memorial,")) == (i, i + 16)
    assert snap_to_words("at Elephant Park", 3, 16) == (3, 16)


@pytest.mark.parametrize(
    ("text", "stage"),
    [
        ("Road operator update: The collision on Park Lane has now been cleared.", "cleared"),
        ("INCIDENT INC-123 CLOSED: debris near Old St. All lanes have reopened.", "cleared"),
        ("finally moving again on City Road after a stranded van #breakdown", "cleared"),
        ("A shunt near Apsley House is GONE, normal service has resumed", "cleared"),
        ("Stand down INC-9: flooding at the Strand resolved. Delays are easing.", "cleared"),
        ("lanes closed while it is cleared, long queues", "active"),
        ("crash on Piccadilly, not cleared yet, police on scene", "active"),
        ("recovery truck on the way, road will be reopened by 6pm", "active"),
        ("serious crash on the Strand, all lanes closed", "active"),
    ],
)
def test_report_stage_detects_clearing_reports(text, stage) -> None:
    from ml.nlp.stage import report_stage

    assert report_stage(text) == stage
