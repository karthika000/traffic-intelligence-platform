"""Prompt templates, versioned as files in ml/summarize/prompts/<version>.txt. The recorded
prompt version includes a content hash, so an edited file can't pass for the old version.

Two layouts:
- seq2seq (v1): "### headline" and "### summary" sections, one generation each
- chat (v2): "### system" and "### user" sections, one chat generation that returns
  "Headline: ..." and "Summary: ..."
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT = "v1"
LAYOUTS = {"seq2seq": {"headline", "summary"}, "chat": {"system", "user"}}


@dataclass(frozen=True)
class PromptTemplate:
    name: str  # e.g. "v1"
    sections: dict[str, str] = field(hash=False)
    digest: str = ""

    @property
    def version(self) -> str:
        return f"{self.name}-{self.digest}"

    @property
    def layout(self) -> str:
        return "chat" if "system" in self.sections else "seq2seq"

    @property
    def headline(self) -> str:
        return self.sections["headline"]

    @property
    def summary(self) -> str:
        return self.sections["summary"]

    def render(self, part: str, facts: str) -> str:
        return self.sections[part].replace("{facts}", facts).strip()

    def messages(self, facts: str) -> list[dict]:
        """Chat layout: system and user messages."""
        return [
            {"role": "system", "content": self.render("system", facts)},
            {"role": "user", "content": self.render("user", facts)},
        ]

    @classmethod
    def load(cls, name: str = DEFAULT_PROMPT) -> "PromptTemplate":
        raw = (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
        sections: dict[str, list[str]] = {}
        current = None
        for line in raw.splitlines():
            if line.startswith("### "):
                current = line[4:].strip()
                sections[current] = []
            elif current is not None:
                sections[current].append(line)
        layout = "chat" if "system" in sections else "seq2seq"
        missing = LAYOUTS[layout] - set(sections)
        if missing:
            raise ValueError(f"prompt {name}: missing sections {sorted(missing)}")
        return cls(
            name=name,
            sections={k: "\n".join(v) for k, v in sections.items()},
            digest=hashlib.sha256(raw.encode()).hexdigest()[:8],
        )
