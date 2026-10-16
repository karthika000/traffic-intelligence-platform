"""Match a report to a camera with the gazetteers the event simulator writes reports from
(traffic_common.gazetteer): official names, streets, landmarks, abbreviations, common
misspellings and area hashtags. Holdout forms are deliberately not used.

Both sides are normalised the same way (lowercase, "&" -> "and", abbreviations expanded:
St -> street, Sq -> square, Cnr/Crnr -> corner, ...; hashtags split into words), then each
alias is compared with every window of about as many words in the text using a fuzzy ratio,
so typos and dropped letters still match. Short aliases (HPC, E&C) must match exactly.

Confidence is the best alias score, reduced when a second camera scores almost as well."""

import re
import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from ml.nlp.labels import split_hashtag
from traffic_common.gazetteer import CameraInfo, gazetteer_for

ABBREVIATIONS = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "sq": "square",
    "ln": "lane",
    "rbt": "roundabout",
    "rdbt": "roundabout",
    "pk": "park",
    "cnr": "corner",
    "gt": "great",
    "cswy": "causeway",
    "stn": "station",
    "x": "cross",
}
_STOP = {"the"}
_TOKEN = re.compile(r"#?[A-Za-z0-9]+(?:'[A-Za-z]+)?|&")


@dataclass(frozen=True)
class Token:
    norm: str
    start: int
    end: int


def tokens(text: str) -> list[Token]:
    """Normalised word tokens with their character offsets in ``text``."""
    text = unicodedata.normalize("NFKC", text)
    out = []
    for m in _TOKEN.finditer(text):
        raw = m.group(0)
        if raw == "&":
            out.append(Token("and", m.start(), m.end()))
            continue
        if raw.startswith("#"):  # #HydeParkCorner -> hyde park corner (offsets: the whole tag)
            for part in split_hashtag(raw[1:]).split():
                out += _norm_token(part, m.start(), m.end())
            continue
        out += _norm_token(raw, m.start(), m.end())
    return out


def _norm_token(raw: str, start: int, end: int) -> list[Token]:
    word = raw.lower().replace("'s", "s").replace("'", "")
    word = ABBREVIATIONS.get(word, word)
    return [] if word in _STOP else [Token(word, start, end)]


def normalize(text: str) -> str:
    return " ".join(t.norm for t in tokens(text))


@dataclass(frozen=True)
class Alias:
    camera_id: str
    text: str  # as written in the gazetteer
    norm: str
    kind: str  # official | street | landmark | abbreviation | misspelling | hashtag | name


@dataclass
class MatchResult:
    camera_id: str | None
    confidence: float
    score: float = 0.0
    matched_text: str | None = None
    alias: str | None = None
    alias_kind: str | None = None
    start: int | None = None
    end: int | None = None
    candidates: list[tuple[str, float]] = field(default_factory=list)  # best per camera

    def to_details(self) -> dict:
        return {
            "camera_score": round(self.score, 4),
            "matched_text": self.matched_text,
            "alias": self.alias,
            "alias_kind": self.alias_kind,
            "candidates": [[c, round(s, 4)] for c, s in self.candidates[:3]],
        }


class CameraMatcher:
    def __init__(
        self,
        cameras: list[CameraInfo],
        min_score: float = 0.85,
        min_confidence: float = 0.7,
        short_alias_chars: int = 5,
    ) -> None:
        self.min_score, self.min_confidence = min_score, min_confidence
        self.short_alias_chars = short_alias_chars
        self.camera_ids = [c.id for c in cameras]
        candidates: dict[str, set[Alias]] = {}
        for cam in cameras:
            gaz = gazetteer_for(cam)
            forms = [
                *((n, "official") for n in gaz.official),
                *((n, "street") for n in gaz.streets),
                *((n, "landmark") for n in gaz.landmarks),
                *((n, "abbreviation") for n in gaz.abbreviations),
                *((n, "misspelling") for n in gaz.misspellings),
                (split_hashtag(gaz.hashtag), "hashtag"),
                (cam.name, "name"),
            ]
            for text, kind in forms:
                norm = normalize(text)
                if norm:
                    candidates.setdefault(norm, set()).add(Alias(cam.id, text, norm, kind))
        # A form two cameras share identifies neither.
        self.aliases = [
            next(iter(group))
            for group in candidates.values()
            if len({a.camera_id for a in group}) == 1
        ]

    def match(self, text: str) -> MatchResult:
        toks = tokens(text)
        windows: dict[int, tuple[list[str], list[tuple[int, int]]]] = {}

        def window(n: int):
            if n not in windows:
                spans = [(i, i + n) for i in range(0, max(0, len(toks) - n + 1))]
                windows[n] = ([" ".join(t.norm for t in toks[a:b]) for a, b in spans], spans)
            return windows[n]

        best: dict[str, tuple[float, Alias, int, int]] = {}
        for alias in self.aliases:
            n = alias.norm.count(" ") + 1
            short = len(alias.norm) < self.short_alias_chars
            for size in (n,) if short else (n - 1, n, n + 1):
                if size < 1:
                    continue
                strings, spans = window(size)
                if not strings:
                    continue
                if short:
                    hits = [i for i, s in enumerate(strings) if s == alias.norm]
                    found = (alias.norm, 100.0, hits[0]) if hits else None
                else:
                    cutoff = 90 if n == 1 else self.min_score * 100
                    found = process.extractOne(
                        alias.norm, strings, scorer=fuzz.ratio, score_cutoff=cutoff
                    )
                if not found:
                    continue
                score = found[1] / 100
                a, b = spans[found[2]]
                current = best.get(alias.camera_id)
                # Prefer the higher score; on a tie, the longer (more specific) alias.
                if current is None or (score, len(alias.norm)) > (
                    current[0],
                    len(current[1].norm),
                ):
                    best[alias.camera_id] = (score, alias, toks[a].start, toks[b - 1].end)
        ranked = sorted(((c, v[0]) for c, v in best.items()), key=lambda cv: -cv[1])
        if not ranked:
            return MatchResult(None, 0.0)
        cam, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = score - second
        confidence = score if margin >= 0.1 else score * (0.6 + 4 * margin)
        _, alias, start, end = best[cam]
        result = MatchResult(
            cam if confidence >= self.min_confidence and score >= self.min_score else None,
            round(confidence, 4),
            score,
            text[start:end],
            alias.text,
            alias.kind,
            start,
            end,
            ranked,
        )
        return result

    def gazetteer_entities(self, text: str, result: MatchResult | None = None) -> list[dict]:
        """The matched span as a location entity (empty if nothing matched)."""
        result = result or self.match(text)
        if result.start is None or result.score < self.min_score:
            return []
        return [
            {
                "text": text[result.start : result.end],
                "label": "LOCATION",
                "start": result.start,
                "end": result.end,
            }
        ]
