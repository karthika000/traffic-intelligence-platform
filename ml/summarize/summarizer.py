"""Model output if it passes the faithfulness guard, the deterministic template otherwise.
Every result records which path was used and why."""

import logging
import re
import time
from dataclasses import asdict, dataclass, field

from ml.summarize import faithfulness
from ml.summarize.context import IncidentContext
from ml.summarize.generator import ChatGenerator, GenerationTimeout, Seq2SeqGenerator
from ml.summarize.prompts import PromptTemplate
from ml.summarize.template import template_headline, template_summary

log = logging.getLogger(__name__)
TEMPLATE_NAME, TEMPLATE_VERSION = "incident-summary-template", "template-1"


@dataclass
class SummarizerConfig:
    generator: str = "model"  # "model" | "template"
    model_name: str = "google/flan-t5-base"
    prompt: str = "v1"
    architecture: str = "seq2seq"  # "seq2seq" (flan-t5, prompt v1) | "chat" (Qwen, prompt v2)
    max_new_tokens_chat: int = 160
    timeout_seconds: float = 20.0
    num_beams: int = 2
    max_new_tokens_summary: int = 120
    max_new_tokens_headline: int = 24

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SummaryResult:
    headline: str
    summary: str
    generation: str  # "model" | "template"
    model_name: str
    model_version: str
    prompt_version: str | None
    latency_ms: float
    faithfulness: dict = field(default_factory=dict)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def tidy(headline: str, summary: str, ctx: IncidentContext) -> tuple[str, str, list[str]]:
    """Enforce the output shape: headline under 15 words, 2-4 sentences ending with the
    (deterministic) recommended action. Returns what was changed."""
    notes = []
    words = headline.replace("\n", " ").strip().strip('"').split()
    if len(words) > 14:
        words, _ = words[:14], notes.append("headline too long")
    headline = " ".join(words).rstrip(".")
    parts = _sentences(summary)
    action = f"Recommended action: {ctx.recommended_action}."
    if not any("recommend" in s.lower() for s in parts):
        parts = parts[:3] + [action]
        notes.append("recommended action appended")
    if len(parts) > 4:
        parts = parts[:3] + [parts[-1]]
        notes.append("trimmed to 4 sentences")
    return headline, " ".join(parts), notes


class IncidentSummarizer:
    def __init__(self, config: SummarizerConfig, generator: Seq2SeqGenerator | None = None):
        self.config = config
        self.prompt = PromptTemplate.load(config.prompt) if config.generator == "model" else None
        if config.generator == "model" and generator is None:
            if config.architecture == "chat":
                generator = ChatGenerator(
                    config.model_name, self.prompt, config.max_new_tokens_chat
                )
            else:
                generator = Seq2SeqGenerator(
                    config.model_name,
                    self.prompt,
                    config.max_new_tokens_headline,
                    config.max_new_tokens_summary,
                    config.num_beams,
                )
        self.generator = generator if config.generator == "model" else None

    @property
    def name(self) -> str:
        return self.config.model_name if self.generator else TEMPLATE_NAME

    def template(
        self, ctx: IncidentContext, started: float, reason: str, extra=None
    ) -> SummaryResult:
        return SummaryResult(
            headline=template_headline(ctx),
            summary=template_summary(ctx),
            generation="template",
            model_name=TEMPLATE_NAME,
            model_version=TEMPLATE_VERSION,
            prompt_version=None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            faithfulness={"passed": True, "fallback_reason": reason, **(extra or {})},
        )

    def summarize(self, ctx: IncidentContext, timeout_s: float | None = None) -> SummaryResult:
        started = time.perf_counter()
        if self.generator is None:
            return self.template(ctx, started, "template configuration")
        timeout = timeout_s if timeout_s is not None else self.config.timeout_seconds
        try:
            raw_headline, raw_summary = self.generator.generate(ctx, timeout)
        except GenerationTimeout as exc:
            return self.template(ctx, started, f"timeout: {exc}")
        except Exception as exc:  # model missing, out of memory, ...
            log.exception("generation failed")
            return self.template(ctx, started, f"error: {type(exc).__name__}: {exc}"[:300])
        headline, summary, notes = tidy(raw_headline, raw_summary, ctx)
        verdict = faithfulness.check(f"{headline}\n{summary}", ctx)
        model_output = {"headline": raw_headline, "summary": raw_summary}
        if not headline or len(raw_summary.split()) < 6:
            return self.template(
                ctx, started, "model output too short", {"model_output": model_output}
            )
        if not verdict.passed:
            return self.template(
                ctx,
                started,
                "faithfulness check failed",
                {"model_violations": verdict.violations, "model_output": model_output},
            )
        # Faithful but not usable for operators: cut-off headline, or type / place missing.
        checks = {
            "headline too long": "headline too long" in notes,
            "headline doesn't name the incident type": not faithfulness.mentions_type(
                headline, ctx
            ),
            "location not named": not faithfulness.mentions_location(f"{headline} {summary}", ctx),
        }
        incomplete = [problem for problem, bad in checks.items() if bad]
        if incomplete:
            return self.template(
                ctx,
                started,
                "model output incomplete: " + "; ".join(incomplete),
                {"model_output": model_output},
            )
        return SummaryResult(
            headline=headline,
            summary=summary,
            generation="model",
            model_name=self.config.model_name,
            model_version=self.generator.revision,
            prompt_version=self.prompt.version if self.prompt else None,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            faithfulness={**verdict.to_dict(), "adjustments": notes},
        )
