"""Faithfulness guard: every number, time, place name and vehicle type in a generated summary
must appear in the facts, and the incident type, severity and status must not be contradicted.
Anything it can't verify counts as a violation (the template is used instead)."""

import re
from dataclasses import dataclass, field

from ml.nlp.entities import VEHICLE, rule_entities
from ml.nlp.matching import normalize
from ml.nlp.stage import report_stage
from ml.summarize.context import IncidentContext, singular
from traffic_common.gazetteer import GAZETTEERS

NUMBER = re.compile(r"(?<![\w.])\d+(?:[.:]\d+)?(?![\w])")
NUMBER_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "dozen": 12,
    "dozens": 24,
    "hundred": 100,
}
STREET = re.compile(
    r"\b(?:[A-Z][a-z']+\s){1,3}(?:Street|St|Road|Rd|Lane|Avenue|Ave|Square|Sq|Bridge|Circus|"
    r"Corner|Place|Way|Hill|Gate|Park|Station|Roundabout|Embankment|Row|Crescent)\b"
)
TYPE_WORDS = {
    "accident": r"crash|collision|accident|pile-?up|rtc",
    "breakdown": r"broken[- ]down|breakdown|stalled|stranded|run out of fuel",
    "road_works": r"road ?works|roadworks|resurfacing|contraflow",
    "weather_hazard": r"flood\w*|ice|icy|snow|fog|storm|surface water|standing water",
    "debris": r"debris|spillage|shed (?:its|a) load|fallen load",
    "police_activity": r"police (?:operation|activity|incident|cordon)|cordon|checkpoint",
}
COMPATIBLE_TYPES = {  # mentioning these alongside the incident type is not a contradiction
    "accident": {"debris", "breakdown"},
    "breakdown": {"accident"},
    "weather_hazard": {"accident", "debris"},
    "debris": {"accident", "breakdown"},
    "police_activity": {"accident"},
    "road_works": set(),
    "congestion": set(),
}
SEVERITY_WORDS = {
    "low": r"low[- ]severity|severity (?:is )?low|minor",
    "medium": r"medium[- ]severity|severity (?:is )?medium",
    "high": r"high[- ]severity|severity (?:is )?high|serious|major|severe",
    "critical": r"critical",
}
SEVERITY_OK = {  # severity words that don't contradict each severity
    "low": {"low"},
    "medium": {"medium", "low"},
    "high": {"high"},
    "critical": {"critical", "high"},
}
# Claims the facts never contain (no source reports casualties, arrests, fires, ...): any
# mention is invented unless the facts say it.
UNSUPPORTED_CLAIMS = (
    r"injur\w*|hurt|wounded|killed|dead|deaths?|died|fatal\w*|casualt\w*|trapped|"
    r"hospital\w*|arrest\w*|fire|blaze|explosion|evacuat\w*|airlift\w*"
)
ALL_PLACES = sorted(
    {
        normalize(name)
        for gaz in GAZETTEERS.values()
        for name in (*gaz.official, *gaz.streets, *gaz.landmarks, *gaz.abbreviations)
        if len(normalize(name)) >= 4
    },
    key=len,
    reverse=True,
)


@dataclass
class Verdict:
    passed: bool
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "violations": self.violations}


def _numbers(text: str) -> set[str]:
    out = set()
    for raw in NUMBER.findall(text):
        out.add(raw)
        if ":" not in raw:
            value = float(raw)
            out.add(f"{value:g}")
            out.add(str(round(value)))
    return out


def check(output: str, ctx: IncidentContext) -> Verdict:
    facts = ctx.facts_text()
    facts_lower, facts_norm = facts.lower(), normalize(facts)
    allowed_numbers = _numbers(facts)
    violations = []

    for raw in NUMBER.findall(output):
        normal = raw if ":" in raw else f"{float(raw):g}"
        if raw not in allowed_numbers and normal not in allowed_numbers:
            violations.append(f"number {raw} not in facts")
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", output, re.I) and str(value) not in allowed_numbers:
            if not re.search(rf"\b{word}\b", facts, re.I):
                violations.append(f"number '{word}' not in facts")
    if re.search(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", output, re.I):
        violations.append("time in am/pm form (facts use 24-hour times)")

    out_norm = f" {normalize(output)} "
    for place in ALL_PLACES:
        if f" {place} " in out_norm and place not in facts_norm:
            violations.append(f"place '{place}' not in facts")
    for m in STREET.finditer(output):
        if normalize(m.group(0)) not in facts_norm:
            violations.append(f"place '{m.group(0)}' not in facts")

    allowed_vehicles = {singular(v) for v in ctx.vehicles}
    for e in rule_entities(output):
        if e["label"] == VEHICLE:
            v = singular(e["text"])
            if v not in allowed_vehicles and v not in facts_lower:
                violations.append(f"vehicle '{e['text']}' not in facts")

    for t, pattern in TYPE_WORDS.items():
        if t == ctx.incident_type or t in COMPATIBLE_TYPES.get(ctx.incident_type, set()):
            continue
        hit = re.search(rf"\b(?:{pattern})\b", output, re.I)
        if hit and hit.group(0).lower() not in facts_lower:
            violations.append(
                f"mentions {t.replace('_', ' ')} ('{hit.group(0)}'), incident is "
                f"{ctx.incident_type.replace('_', ' ')}"
            )
    for level, pattern in SEVERITY_WORDS.items():
        if level not in SEVERITY_OK.get(ctx.severity, {ctx.severity}) and re.search(
            rf"(?<![\w-])(?:{pattern})(?![\w-])", output, re.I
        ):
            violations.append(f"severity '{level}' contradicts {ctx.severity}")
    if ctx.status != "resolved" and report_stage(output) == "cleared":
        violations.append(f"says it is over, but status is {ctx.status}")
    if ctx.status == "resolved" and re.search(
        r"\b(?:ongoing|still blocked|is blocking)\b", output, re.I
    ):
        violations.append("says it is ongoing, but status is resolved")
    for hit in re.finditer(rf"\b(?:{UNSUPPORTED_CLAIMS})\b", output, re.I):
        if hit.group(0).lower() not in facts_lower:
            violations.append(f"unsupported claim '{hit.group(0)}'")
    return Verdict(not violations, sorted(set(violations)))


def mentions_type(text: str, ctx: IncidentContext) -> bool:
    """Whether the text names the incident type (its label or a keyword for it)."""
    words = [w for w in re.split(r"[\s-]+", ctx.type_label.lower()) if len(w) > 3]
    if any(re.search(rf"\b{re.escape(w)}", text, re.I) for w in words):
        return True
    pattern = TYPE_WORDS.get(ctx.incident_type)
    return bool(pattern and re.search(rf"\b(?:{pattern})\b", text, re.I))


def mentions_location(text: str, ctx: IncidentContext) -> bool:
    return normalize(ctx.short_location) in normalize(text)
