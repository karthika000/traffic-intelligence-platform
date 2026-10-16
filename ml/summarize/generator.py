"""Local generation with a per-incident time budget. Weights come from the Hugging Face cache
(HF_HOME, a shared Docker volume); nothing calls an external API.

- Seq2SeqGenerator: flan-t5 style models, one generation for the headline and one for the
  summary (seq2seq prompt layout, e.g. v1)
- ChatGenerator: instruction-tuned chat models (e.g. Qwen2.5-Instruct), one generation that
  returns "Headline: ..." and "Summary: ..." (chat prompt layout, e.g. v2)
"""

import logging
import re
import time
from pathlib import Path

from ml.summarize.context import IncidentContext
from ml.summarize.prompts import PromptTemplate

log = logging.getLogger(__name__)


class GenerationTimeout(TimeoutError):
    pass


def _revision(model, model_name: str) -> str:
    """The Hub commit the weights came from (from the config, or the cache snapshot dir)."""
    commit = getattr(model.config, "_commit_hash", None)
    if not commit:
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model_name, "config.json")
            commit = Path(cached).parent.name if isinstance(cached, str) else None
        except Exception:
            commit = None
    return commit[:12] if commit else "unknown"


class _Generator:
    auto_class = "AutoModelForSeq2SeqLM"

    def __init__(self, model_name: str, prompt: PromptTemplate, num_beams: int = 1) -> None:
        self.model_name, self.prompt, self.num_beams = model_name, prompt, num_beams
        self._model = self._tokenizer = None
        self.revision = "unknown"

    def load(self):
        if self._model is None:
            import transformers

            started = time.perf_counter()
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            auto = getattr(transformers, self.auto_class)
            self._model = auto.from_pretrained(self.model_name).eval()
            self.revision = _revision(self._model, self.model_name)
            log.info(
                "loaded %s (%s) in %.1fs",
                self.model_name,
                self.revision,
                time.perf_counter() - started,
            )
        return self

    def _run(self, inputs: dict, max_new_tokens: int, deadline: float, what: str):
        import torch

        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise GenerationTimeout(f"no time left for the {what}")
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                no_repeat_ngram_size=3,
                max_time=remaining,  # stops generating when the budget is used up
            )
        if time.perf_counter() > deadline:
            raise GenerationTimeout(f"{what} took longer than the time limit")
        return out


class Seq2SeqGenerator(_Generator):
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        prompt: PromptTemplate | None = None,
        max_new_tokens_headline: int = 24,
        max_new_tokens_summary: int = 120,
        num_beams: int = 2,
    ) -> None:
        super().__init__(model_name, prompt or PromptTemplate.load("v1"), num_beams)
        self.max_new_tokens = {
            "headline": max_new_tokens_headline,
            "summary": max_new_tokens_summary,
        }

    def _part(self, part: str, facts: str, deadline: float) -> str:
        inputs = self._tokenizer(
            self.prompt.render(part, facts), return_tensors="pt", truncation=True, max_length=512
        )
        out = self._run(dict(inputs), self.max_new_tokens[part], deadline, part)
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def generate(self, ctx: IncidentContext, timeout_s: float) -> tuple[str, str]:
        """(headline, summary) for the context, or GenerationTimeout."""
        self.load()
        deadline = time.perf_counter() + timeout_s
        facts = ctx.facts_text()
        return self._part("headline", facts, deadline), self._part("summary", facts, deadline)


_HEADLINE = re.compile(r"^\W*headline\W*:\s*(.+)$", re.I | re.M)
_SUMMARY = re.compile(r"^\W*summary\W*:\s*(.+)", re.I | re.M | re.S)


def parse_chat_output(text: str) -> tuple[str, str]:
    """'Headline: ...' / 'Summary: ...' -> (headline, summary); first line and the rest if the
    model ignored the format."""
    h, s = _HEADLINE.search(text), _SUMMARY.search(text)
    if h and s:
        return h.group(1).strip(" *\"'"), " ".join(s.group(1).strip(" *").split())
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "", ""
    return lines[0].strip("*#\"' "), " ".join(" ".join(lines[1:]).split())


class ChatGenerator(_Generator):
    auto_class = "AutoModelForCausalLM"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        prompt: PromptTemplate | None = None,
        max_new_tokens: int = 160,
    ) -> None:
        super().__init__(model_name, prompt or PromptTemplate.load("v2"), num_beams=1)
        if self.prompt.layout != "chat":
            raise ValueError(f"{model_name} needs a chat prompt (system/user), not {prompt.name}")
        self.max_new_tokens = max_new_tokens

    def generate(self, ctx: IncidentContext, timeout_s: float) -> tuple[str, str]:
        self.load()
        deadline = time.perf_counter() + timeout_s
        inputs = self._tokenizer.apply_chat_template(
            self.prompt.messages(ctx.facts_text()),
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        out = self._run(dict(inputs), self.max_new_tokens, deadline, "briefing")
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return parse_chat_output(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
