"""Rule-based entities (vehicle types, lane information, times) and merging them with location
spans from NER and the gazetteer. Entities are plain dicts shaped like
traffic_common.schemas.Entity: {"text", "label", "start", "end"}."""

import re

VEHICLE, LANE, TIME, LOCATION = "VEHICLE", "LANE", "TIME", "LOCATION"

_NUM = r"(?:one|two|three|four|1|2|3|4)"
_PATTERNS = {
    VEHICLE: [
        r"articulated lorr(?:y|ies)|tow truck|recovery (?:truck|vehicle)|fire engines?",
        r"police (?:cars?|vans?)|delivery vans?|black cabs?|double[- ]deckers?|cement mixers?",
        r"lorr(?:y|ies)|hgvs?|trucks?|tankers?|vans?|cars?|buse?s|bus|coach(?:es)?|minibus",
        r"taxis?|cabs?|motorbikes?|motorcycles?|mopeds?|scooters?|bicycles?|bikes?",
        r"cyclists?|ambulances?|suvs?|pick-?ups?|tractors?|skip lorry",
    ],
    LANE: [
        rf"(?:lanes?|ln) {_NUM}",
        rf"(?:{_NUM}|all|both) (?:of the )?lanes?(?: (?:closed|shut|blocked|open))?",
        r"(?:left|right|middle|centre|center|inside|outside|offside|nearside|bus|cycle"
        r"|turning|slip) lanes?",
        r"hard shoulder|contraflow|(?:north|south|east|west)bound(?: carriageway)?",
    ],
    TIME: [
        r"\d{1,2}[:.]\d{2}\s?(?:am|pm)?|\d{1,2}\s?(?:am|pm)",
        r"(?:this|tomorrow|yesterday) (?:morning|afternoon|evening)|tonight|rush hour",
        r"(?:for|since|about|around|over) (?:the )?(?:last |past )?(?:\d+|an?|half an)"
        r" ?(?:mins?|minutes|hours?|hrs?)",
        r"\d+ ?(?:mins?|minutes|hrs?|hours)|half an hour",
    ],
}
_COMPILED = {
    label: re.compile(rf"(?<![\w-])(?:{'|'.join(parts)})(?![\w-])", re.IGNORECASE)
    for label, parts in _PATTERNS.items()
}


def rule_entities(text: str) -> list[dict]:
    """Vehicle types, lane information and times found by regular expressions."""
    found = [
        {"text": m.group(0), "label": label, "start": m.start(), "end": m.end()}
        for label, rx in _COMPILED.items()
        for m in rx.finditer(text)
    ]
    return resolve_overlaps(found)


def resolve_overlaps(entities: list[dict]) -> list[dict]:
    """Keep the longest of overlapping spans (earlier wins a tie), sorted by position."""
    kept: list[dict] = []
    for e in sorted(entities, key=lambda e: (-(e["end"] - e["start"]), e["start"])):
        if all(e["end"] <= k["start"] or e["start"] >= k["end"] for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e["start"])


def location_spans(text: str, ner: list[dict], gazetteer: list[dict]) -> list[dict]:
    """Location entities: NER LOC spans, NER ORG/MISC spans the gazetteer confirms (overlap),
    and gazetteer matches NER missed (it misses most lowercase driver and social posts)."""

    def overlaps(a: dict, b: dict) -> bool:
        return a["start"] < b["end"] and b["start"] < a["end"]

    spans = []
    for e in ner:
        if e["label"] == "LOC" or any(overlaps(e, g) for g in gazetteer):
            spans.append(
                {
                    "text": text[e["start"] : e["end"]],
                    "label": LOCATION,
                    "start": e["start"],
                    "end": e["end"],
                }
            )
    spans += [{**g, "label": LOCATION} for g in gazetteer]
    return resolve_overlaps(spans)
