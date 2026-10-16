"""Text classifier training and packaging, offline: a tiny randomly initialised DistilBERT
(built here, nothing downloaded) and the TF-IDF baseline, each saved and reloaded through
the pyfunc wrapper. Needs PyTorch and transformers (the tests image has them)."""

import itertools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")
pytest.importorskip("transformers")
pd = pytest.importorskip("pandas")

from ml.nlp.baseline import TfidfBaseline  # noqa: E402
from ml.nlp.labels import TASKS, label_maps  # noqa: E402
from ml.nlp.predict import OUTPUT_COLUMNS  # noqa: E402
from ml.nlp.pyfunc import IncidentTextClassifier, save_baseline_dir  # noqa: E402

PLACES = ["Hyde Park Corner", "Old Street", "Trafalgar Square", "the Strand"]
PATTERNS = [
    ("serious crash involving a {v} at {p}, all lanes closed", "accident", "critical"),
    ("long queues near {p}, slow traffic", "congestion", "medium"),
    ("flooding at {p}, standing water", "weather_hazard", "high"),
    ("broken down {v} on {p}, recovery on the way", "breakdown", "medium"),
    ("lovely morning, nothing to report near {p}", "none", "low"),
]
VEHICLES = ["lorry", "van", "bus"]


def dataset() -> pd.DataFrame:
    rows = [
        {"text": t.format(v=v, p=p), "incident_type": it, "severity": sev}
        for (t, it, sev), v, p in itertools.product(PATTERNS, VEHICLES, PLACES)
    ]
    return pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)


def labels(df: pd.DataFrame):
    maps = label_maps()
    return {t: maps[t].encode(df[t]) for t in TASKS}


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory) -> Path:
    """A 1-layer DistilBERT with a vocabulary of the dataset's words, saved like a Hub model."""
    from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

    from ml.nlp.labels import preprocess

    words = sorted(
        {w for t in dataset().text for w in preprocess(t).lower().replace(",", " ,").split()}
    )
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    tokenizer = DistilBertTokenizer(vocab={w: i for i, w in enumerate(vocab)})
    config = DistilBertConfig(
        vocab_size=len(vocab),
        dim=32,
        hidden_dim=64,
        n_layers=1,
        n_heads=2,
        max_position_embeddings=64,
    )
    out = tmp_path_factory.mktemp("tiny-distilbert")
    DistilBertModel(config).save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


def test_transformer_learns_saves_and_reloads(tiny_base: Path, tmp_path: Path) -> None:
    from ml.nlp import transformer

    df = dataset()
    y = labels(df)
    cfg = transformer.TransformerConfig(
        base_model=str(tiny_base),
        max_length=32,
        epochs=12,
        batch_size=8,
        learning_rate=3e-3,
        warmup_ratio=0.05,
        seed=1,
    )
    epochs_seen = []
    model, tokenizer, history = transformer.train(
        df.text.tolist(),
        y,
        df.text.tolist(),
        y,
        {t: len(label_maps()[t]) for t in TASKS},
        cfg,
        on_epoch=lambda e, h: epochs_seen.append(e),
    )
    assert epochs_seen == list(range(1, 13)) and len(history) == 12
    best = max(h["val_mean_macro_f1"] for h in history)
    assert best >= 0.8, history[-1]  # an easy, separable dataset
    assert history[-1]["train_loss"] < history[0]["train_loss"]

    before = transformer.predict_proba(model, tokenizer, df.text.tolist(), cfg.max_length)
    model_dir = transformer.save(model, tokenizer, cfg, label_maps(), tmp_path / "model")
    assert json.loads((model_dir / "config.json").read_text())["kind"] == "transformer"
    loaded = IncidentTextClassifier().load_dir(model_dir)
    out = loaded.predict(None, pd.DataFrame({"text": df.text}))
    assert list(out.columns) == OUTPUT_COLUMNS and len(out) == len(df)
    after = loaded.predict_probabilities(df.text.tolist())
    for task in TASKS:  # the reloaded model predicts exactly what the trained one did
        assert abs(after[task] - before[task]).max() < 1e-5
    probs = json.loads(out.probabilities[0])
    assert abs(sum(probs["severity"].values()) - 1) < 1e-3
    assert loaded.predict(None, ["crash on the strand"]).shape == (1, len(OUTPUT_COLUMNS))


def test_baseline_pyfunc_round_trip(tmp_path: Path) -> None:
    df = dataset()
    model = TfidfBaseline(seed=0).fit(df.text.tolist(), labels(df))
    loaded = IncidentTextClassifier().load_dir(save_baseline_dir(model, label_maps(), tmp_path))
    out = loaded.predict(None, pd.DataFrame({"text": df.text}))
    assert (out.incident_type == df.incident_type).mean() == 1.0
    assert (out.severity == df.severity).mean() == 1.0
    assert out.incident_confidence.between(0, 1).all()
    unseen = loaded.predict(None, ["serious crash involving a coach at Old Street"])
    assert unseen.incident_type[0] == "accident"
