"""DistilBERT with two classification heads (incident type, severity) on one shared encoder:
one forward pass per report, half the CPU cost of two fine-tuned models.

Imports PyTorch and transformers at module level; only training and the pyfunc load it."""

import copy
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ml.nlp.labels import TASKS, LabelMap, preprocess
from ml.nlp.predict import softmax

log = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    base_model: str = "distilbert-base-uncased"
    max_length: int = 64  # reports are short: ~16 words on average, 35 at most
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    severity_loss_weight: float = 1.0
    seed: int = 13


class MultiTaskClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_labels: dict[str, int], dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, n) for task, n in n_labels.items()})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])  # [CLS] token
        return {task: head(cls) for task, head in self.heads.items()}


def encode(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    """Tokenise one batch, padded to its longest text (not max_length): much faster on CPU."""
    enc = tokenizer(
        [preprocess(t) for t in texts],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_proba(
    model: MultiTaskClassifier, tokenizer, texts: list[str], max_length: int, batch_size: int = 64
) -> dict[str, np.ndarray]:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {task: [] for task in model.heads}
    for i in range(0, len(texts), batch_size):
        logits = model(**encode(tokenizer, texts[i : i + batch_size], max_length))
        for task, value in logits.items():
            chunks[task].append(softmax(value.float().numpy()))
    n_labels = {task: head.out_features for task, head in model.heads.items()}
    return {
        task: np.concatenate(parts) if parts else np.zeros((0, n_labels[task]))
        for task, parts in chunks.items()
    }


def _scores(probs: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for task in TASKS:
        pred = probs[task].argmax(axis=1)
        out[f"{task}_accuracy"] = float(accuracy_score(labels[task], pred))
        out[f"{task}_macro_f1"] = float(f1_score(labels[task], pred, average="macro"))
    out["mean_macro_f1"] = float(np.mean([out[f"{t}_macro_f1"] for t in TASKS]))
    return out


def train(
    texts: list[str],
    labels: dict[str, np.ndarray],
    val_texts: list[str],
    val_labels: dict[str, np.ndarray],
    n_labels: dict[str, int],
    cfg: TransformerConfig,
    on_epoch: Callable[[int, dict], None] | None = None,
    init: tuple[MultiTaskClassifier, object] | None = None,
) -> tuple[MultiTaskClassifier, object, list[dict]]:
    """Fine-tune; keeps the epoch with the best validation mean macro-F1. ``init`` (a trained
    model and its tokenizer) warm-starts from that model instead of ``cfg.base_model``."""
    _seed(cfg.seed)
    if init is not None:
        trained, tokenizer = init
        model = MultiTaskClassifier(copy.deepcopy(trained.encoder), n_labels, cfg.dropout)
        model.heads.load_state_dict(trained.heads.state_dict())
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        model = MultiTaskClassifier(
            AutoModel.from_pretrained(cfg.base_model), n_labels, cfg.dropout
        )
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate)
    steps_per_epoch = (len(texts) + cfg.batch_size - 1) // cfg.batch_size
    total = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total * cfg.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            (step + 1) / warmup
            if step < warmup
            else max(0.0, (total - step) / max(1, total - warmup))
        ),
    )
    loss_fn = nn.CrossEntropyLoss()
    weights = {"incident_type": 1.0, "severity": cfg.severity_loss_weight}
    targets = {task: torch.as_tensor(labels[task]) for task in TASKS}
    rng = np.random.default_rng(cfg.seed)
    history, best, best_state = [], -1.0, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        started, running = time.perf_counter(), 0.0
        order = rng.permutation(len(texts))
        for step in range(steps_per_epoch):
            idx = order[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            logits = model(**encode(tokenizer, [texts[i] for i in idx], cfg.max_length))
            loss = sum(weights[t] * loss_fn(logits[t], targets[t][idx]) for t in TASKS)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch, step, steps_per_epoch, loss.item())
        val = _scores(predict_proba(model, tokenizer, val_texts, cfg.max_length), val_labels)
        entry = {
            "epoch": epoch,
            "train_loss": running / steps_per_epoch,
            "epoch_seconds": time.perf_counter() - started,
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(entry)
        log.info("epoch %d: %s", epoch, {k: round(v, 4) for k, v in entry.items()})
        if on_epoch:
            on_epoch(epoch, entry)
        if val["mean_macro_f1"] > best:
            best, best_state = val["mean_macro_f1"], copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, history


def save(
    model: MultiTaskClassifier,
    tokenizer,
    cfg: TransformerConfig,
    maps: dict[str, LabelMap],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    torch.save(model.heads.state_dict(), out_dir / "heads.pt")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "kind": "transformer",
                "transformer": asdict(cfg),
                "label_maps": {t: m.to_dict() for t, m in maps.items()},
            },
            indent=2,
        )
    )
    return out_dir


def load(model_dir: Path) -> tuple[MultiTaskClassifier, object, TransformerConfig]:
    config = json.loads((model_dir / "config.json").read_text())
    cfg = TransformerConfig(**config["transformer"])
    n_labels = {t: len(m["labels"]) for t, m in config["label_maps"].items()}
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = MultiTaskClassifier(AutoModel.from_pretrained(model_dir / "encoder"), n_labels, 0.0)
    model.heads.load_state_dict(torch.load(model_dir / "heads.pt", map_location="cpu"))
    model.eval()
    return model, tokenizer, cfg
