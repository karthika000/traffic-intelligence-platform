"""Synthetic free-text traffic reports with ground-truth labels.

A report is assembled from an incident phrase (``core``), a severity phrase (``detail``),
a location mention and a style template (operator, police, driver, social media), then
style-specific noise is applied: typos, abbreviations, lowercasing, missing punctuation,
emojis, hashtags. The location mention is protected from typos so ``location_mention``
is always an exact substring of the text.

Every phrase pool has ``common`` items and ``holdout`` items. The TRAIN variant never uses
holdout items; TEST uses them about half the time; LIVE uses everything. That way a model
trained on the train split is evaluated on phrasings it has not memorised.
"""

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from event_simulator import slang
from event_simulator.locations import CameraInfo, Gazetteer, gazetteer_for
from traffic_common.enums import IncidentType, ReportSource, Severity

T = IncidentType
S = Severity


class Style(StrEnum):
    DRIVER = "driver"
    POLICE = "police"
    OPERATOR = "operator"
    SOCIAL_MEDIA = "social_media"


STYLE_SOURCE = {
    Style.DRIVER: ReportSource.CITIZEN,
    Style.POLICE: ReportSource.POLICE,
    Style.OPERATOR: ReportSource.OPERATOR,
    Style.SOCIAL_MEDIA: ReportSource.SOCIAL_MEDIA,
}


class Variant(StrEnum):
    TRAIN = "train"
    TEST = "test"
    LIVE = "live"


class Stage(StrEnum):
    ACTIVE = "active"
    CLEARED = "cleared"


@dataclass(frozen=True)
class Pool:
    common: tuple[str, ...]
    holdout: tuple[str, ...] = ()

    def pick(self, rng: random.Random, variant: Variant) -> str:
        if variant is Variant.TRAIN or not self.holdout:
            return rng.choice(self.common)
        if variant is Variant.TEST:
            return rng.choice(self.holdout if rng.random() < 0.5 else self.common)
        return rng.choice(self.common + self.holdout)

    def all(self) -> tuple[str, ...]:
        return self.common + self.holdout


def P(common: Sequence[str], holdout: Sequence[str] = ()) -> Pool:
    return Pool(tuple(common), tuple(holdout))


# --------------------------------------------------------------------------- incident phrases

CORE: dict[IncidentType, Pool] = {
    T.ACCIDENT: P(["a collision", "a crash", "an RTC", "a car accident", "a collision between two cars",
                   "a crash involving a bus", "a motorbike collision", "a vehicle collision"],
                  ["a shunt", "a smash involving a taxi", "an accident involving a cyclist"]),
    T.BREAKDOWN: P(["a broken-down car", "a broken down lorry", "a stalled vehicle", "a breakdown",
                    "a stranded van", "a broken-down bus", "a car with a flat tyre",
                    "a vehicle that has run out of fuel"],
                   ["a conked-out van", "a disabled HGV", "a stalled taxi"]),
    T.ROAD_WORKS: P(["roadworks", "road works", "emergency gas works", "resurfacing works",
                     "temporary traffic lights", "utility works", "water main repairs",
                     "a lane closure for works"],
                    ["cable laying works", "a works convoy", "kerb replacement works"]),
    T.CONGESTION: P(["heavy traffic", "congestion", "long queues", "slow-moving traffic",
                     "stationary traffic", "tailbacks", "a traffic jam", "nose-to-tail traffic"],
                    ["gridlocked traffic", "a crawl", "bumper-to-bumper traffic"]),
    T.WEATHER_HAZARD: P(["surface water", "flooding", "standing water", "ice on the road",
                         "thick fog", "a fallen tree", "black ice", "heavy spray from the rain"],
                        ["a flooded carriageway", "hailstones settling", "strong crosswinds"]),
    T.DEBRIS: P(["debris in the road", "a fallen load", "a mattress in the carriageway", "tyre debris",
                 "broken glass across the lane", "scaffolding poles in the road",
                 "an object in the road", "a shed load"],
                ["a wheelie bin in the road", "loose gravel spilled", "a fallen road sign"]),
    T.POLICE_ACTIVITY: P(["a police incident", "police activity", "a police cordon",
                          "a road closure by police", "officers stopping traffic",
                          "a police checkpoint", "an emergency services incident",
                          "a police vehicle stop"],
                         ["an ongoing police operation", "a security alert", "a cordon by officers"]),
}  # fmt: skip

# Severity is carried by these phrases, so the label is recoverable from the text.
DETAIL: dict[IncidentType, dict[Severity, Pool]] = {
    T.ACCIDENT: {
        S.LOW: P(["minor bump, no injuries, cars already moved aside", "small scrape, traffic still flowing",
                  "drivers swapping details on the kerb, no real delay"], ["just a knock, nobody hurt"]),
        S.MEDIUM: P(["one lane blocked, some delays building", "cars stopped in the inside lane, slow going past",
                     "lane partially blocked while vehicles are moved"], ["single lane obstructed, moderate delays"]),
        S.HIGH: P(["two lanes blocked and an ambulance on scene", "injuries reported and long delays",
                   "road partly closed, expect major delays"], ["paramedics attending, queues back a long way"]),
        S.CRITICAL: P(["multiple vehicles involved and the road fully closed",
                       "serious injuries reported, air ambulance requested",
                       "all lanes closed, avoid the area completely", "vehicle overturned with casualties"],
                      ["fire crews cutting people free, road shut"]),
    },
    T.BREAKDOWN: {
        S.LOW: P(["pulled onto the kerb with hazards on, not blocking anything", "already off the road, no delay",
                  "recovery on the way, traffic unaffected"], ["tucked into a bus stop, no impact"]),
        S.MEDIUM: P(["blocking the left lane, slow past it", "stuck in a lane with hazards on, some delays",
                     "partly blocking the junction, traffic squeezing past"],
                    ["half in the lane, cars merging around it"]),
        S.HIGH: P(["blocking two lanes and queues growing fast", "stuck right on the junction, traffic backing up badly",
                   "recovery truck needed, long delays"], ["jammed across the box junction, big queues"]),
        S.CRITICAL: P(["completely blocking the road, nothing getting through",
                       "jackknifed across all lanes, road at a standstill",
                       "smoke pouring out, road closed both ways"], ["leaking fuel everywhere, full closure"]),
    },
    T.ROAD_WORKS: {
        S.LOW: P(["off-peak only, minimal delays", "barriers on the pavement side, traffic moving fine",
                  "works on the verge, no lane closures"], ["small works area, barely noticeable"]),
        S.MEDIUM: P(["one lane closed, some delays at peak times", "temporary lights causing short queues",
                     "lane narrowed, take care"], ["contraflow in place, a few minutes extra"]),
        S.HIGH: P(["two lanes closed and long queues", "lights stuck on red causing big tailbacks",
                   "works overrunning into rush hour, heavy delays"],
                  ["only one lane open, queues back a long way"]),
        S.CRITICAL: P(["road fully closed for emergency works", "no access at all, diversions in place",
                       "a burst main has shut the junction completely"], ["total closure, buses diverted"]),
    },
    T.CONGESTION: {
        S.LOW: P(["a bit slower than usual", "busy but moving", "minor delays only"],
                 ["slightly sluggish this morning"]),
        S.MEDIUM: P(["queues of about 10 minutes", "stop-start traffic, allow extra time",
                     "slow going for a few hundred metres"], ["crawling along for a while"]),
        S.HIGH: P(["queues of over 30 minutes", "traffic barely moving for a mile",
                   "huge tailbacks back to the previous junction"], ["lost 40 minutes sitting here"]),
        S.CRITICAL: P(["complete gridlock, nothing moving at all", "standstill for over an hour",
                       "every approach completely locked up"], ["total gridlock, people turning engines off"]),
    },
    T.WEATHER_HAZARD: {
        S.LOW: P(["a few puddles, nothing serious", "light mist, take a bit of care", "slightly slippery surface"],
                 ["damp patches, drive gently"]),
        S.MEDIUM: P(["water across one lane, slow down", "visibility down to about 200 metres",
                     "slippery surface, cars braking early"], ["spray making it hard to see"]),
        S.HIGH: P(["deep water across the road, cars struggling", "visibility under 50 metres, very dangerous",
                   "a tree blocking a lane, branches everywhere"], ["water up to the wheel arches"]),
        S.CRITICAL: P(["road flooded and impassable, cars stranded", "tree down across all lanes, road closed",
                       "sheet ice, several cars spun out"],
                      ["water rising fast, emergency services rescuing drivers"]),
    },
    T.DEBRIS: {
        S.LOW: P(["off to the side, not really in the way", "small bits on the edge of the lane",
                  "already being swept up"], ["mostly on the verge"]),
        S.MEDIUM: P(["in lane one, cars swerving around it", "scattered over a lane, slow down",
                     "partly blocking the lane"], ["drivers weaving to avoid it"]),
        S.HIGH: P(["across two lanes, traffic stopping suddenly", "cars hitting it and pulling over with damage",
                   "lanes closed while it is cleared, long queues"], ["several punctures already"]),
        S.CRITICAL: P(["spread across every lane, road closed", "caused a crash and the road is shut",
                       "dangerous load spill, full closure"], ["hazardous spill, area evacuated"]),
    },
    T.POLICE_ACTIVITY: {
        S.LOW: P(["officers on the pavement, traffic unaffected", "routine stop, no delays",
                  "a police car parked up, nothing to worry about"], ["low-key patrol, no impact"]),
        S.MEDIUM: P(["one lane coned off by officers, some delays", "police directing traffic, slow going",
                     "vehicles being checked, short queues"], ["officers waving traffic through slowly"]),
        S.HIGH: P(["road partly closed by police, long delays", "large police presence, lanes shut",
                   "cordon in place, expect heavy delays"], ["several police vans blocking lanes"]),
        S.CRITICAL: P(["road completely closed by police, avoid the area", "major incident declared, full cordon",
                       "area evacuated, all roads shut"], ["armed officers on scene, total closure"]),
    },
}  # fmt: skip

CLEARED_DETAIL = P(["all lanes have reopened", "traffic is returning to normal", "the road is clear again",
                    "delays are easing", "recovery is complete"], ["normal service has resumed"])  # fmt: skip

# --------------------------------------------------------------------------- style templates
# Placeholders: {core} {Core} {detail} {Detail} {cleared} {Cleared} {loc} (with preposition)
# {loc_bare} (mention only) {opener} {closer} {emoji} {tags} {time} {ref}

ACTIVE_TEMPLATES: dict[Style, Pool] = {
    Style.OPERATOR: P(["{opener} {Core} {loc}. {Detail}. {closer}",
                       "{time} - {Core} reported {loc}. {Detail}. {closer}",
                       "INCIDENT {ref}: {core} {loc}. {Detail}.",
                       "Traffic alert: {core} {loc}. {Detail}. {closer}",
                       "{Core} {loc} ({ref}). {Detail}. {closer}"],
                      ["Network update {time}: please be aware of {core} {loc}. {Detail}."]),
    Style.POLICE: P(["Officers are dealing with {core} {loc}. {Detail}. {closer}",
                     "We are on scene {loc} following {core}. {Detail}. {closer}",
                     "Police are aware of {core} {loc} - {detail}. {closer}",
                     "{Core} {loc}: {detail}. {closer}"],
                    ["Update from officers {loc}: {core}, {detail}."]),
    Style.DRIVER: P(["{core} {loc}, {detail}", "just passed {core} {loc} - {detail}",
                     "stuck because of {core} {loc}. {detail}", "heads up: {core} {loc}, {detail}",
                     "anyone else seeing {core} {loc}? {detail}", "{core} {loc} right now. {detail}"],
                    ["fyi {core} {loc} {detail}", "be warned, {core} {loc}... {detail}"]),
    Style.SOCIAL_MEDIA: P(["{emoji} {Core} {loc}!! {detail} {tags}", "{core} {loc} {emoji} {detail} {tags}",
                           "{tags} {Core} {loc}... {detail} {emoji}",
                           "Avoid {loc_bare} rn {emoji} {core}, {detail} {tags}",
                           "omg {core} {loc} {emoji}{emoji} {detail} {tags}"],
                          ["can't believe it, {core} {loc} AGAIN {emoji} {detail} {tags}"]),
}  # fmt: skip

CLEARED_TEMPLATES: dict[Style, Pool] = {
    Style.OPERATOR: P(["{opener} {Core} {loc} has now been cleared. {Cleared}.",
                       "{time} - UPDATE: {core} {loc} cleared. {Cleared}.",
                       "INCIDENT {ref} CLOSED: {core} {loc}. {Cleared}."],
                      ["Stand down {ref}: {core} {loc} resolved. {Cleared}."]),
    Style.POLICE: P(["Update: {core} {loc} has been dealt with. {Cleared}. Thanks for your patience.",
                     "{Core} {loc} now cleared. {Cleared}."],
                    ["Officers have left the scene {loc} after {core}. {Cleared}."]),
    Style.DRIVER: P(["{core} {loc} finally cleared, {cleared}", "all good now {loc}, {core} gone and {cleared}"],
                    ["{core} {loc} sorted at last, {cleared}"]),
    Style.SOCIAL_MEDIA: P(["{core} {loc} cleared {emoji} {cleared} {tags}",
                           "finally moving again {loc} after {core} {emoji} {tags}"],
                          ["{Core} {loc} is GONE {emoji} {cleared} {tags}"]),
}  # fmt: skip

CHATTER_TEMPLATES: dict[Style, Pool] = {
    Style.OPERATOR: P(["Routine update: no incidents reported {loc}. Traffic flowing normally.",
                       "All lanes open {loc}. No delays at present.",
                       "Network update {time}: no planned works {loc} today."],
                      ["Scheduled maintenance {loc} has been postponed; no delays expected."]),
    Style.POLICE: P(["Officers will be holding a community meeting {loc} this evening.",
                     "Thank you to everyone who came to our cycle safety event {loc}.",
                     "Reminder: please lock your bikes, especially {loc}."],
                    ["Our neighbourhood team will be on patrol {loc} this weekend."]),
    Style.DRIVER: P(["lovely drive {loc} this morning, all clear", "testing my new dashcam {loc}",
                     "anyone know a decent coffee place {loc}?", "nice to see the road quiet {loc} for once",
                     "running late again lol, nothing to do with traffic"],
                    ["listening to a great podcast while driving {loc}"]),
    Style.SOCIAL_MEDIA: P(["beautiful sunset {loc} {emoji} {tags}", "the buskers {loc} are amazing today {emoji}",
                           "brunch spot {loc} is 10/10 {emoji} {tags}", "who else is at the gig tonight? {emoji}",
                           "sun's out {loc} {emoji} {tags}"],
                          ["pigeons {loc} are absolutely fearless {emoji}"]),
}  # fmt: skip

OPENERS = P(["TfL Traffic:", "Network Control:", "Road operator update:", "ALERT:", "Traffic Control Centre:"],
            ["Highways update:"])  # fmt: skip
OPERATOR_CLOSERS = P(["Please allow extra journey time.", "Avoid the area if possible.",
                      "Diversions are in place.", "Further updates to follow.",
                      "Consider alternative routes."], ["Engineers are on site."])  # fmt: skip
POLICE_CLOSERS = P(["Please avoid the area.", "Thank you for your patience.", "Please do not stop to film.",
                    "Follow directions from officers on scene.", "More information when we have it."],
                   ["Drive carefully around the scene."])  # fmt: skip

TYPE_EMOJI = {
    T.ACCIDENT: ["🚨", "🚑", "💥", "😱"], T.BREAKDOWN: ["🚛", "🔧", "😩", "🙄"],
    T.ROAD_WORKS: ["🚧", "🦺", "🙄", "😤"], T.CONGESTION: ["🐌", "😡", "🚗🚗🚗", "😤"],
    T.WEATHER_HAZARD: ["🌧️", "🌊", "🌫️", "❄️"], T.DEBRIS: ["⚠️", "😬", "🛞", "🙃"],
    T.POLICE_ACTIVITY: ["🚓", "👮", "🚨", "👀"], T.NONE: ["☀️", "😍", "☕", "🎶", "😂"],
}  # fmt: skip
TYPE_TAGS = {
    T.ACCIDENT: ["#crash", "#accident", "#RTC"], T.BREAKDOWN: ["#breakdown", "#stuck"],
    T.ROAD_WORKS: ["#roadworks", "#TempLights"], T.CONGESTION: ["#traffic", "#gridlock", "#commute"],
    T.WEATHER_HAZARD: ["#flooding", "#UKweather", "#fog"], T.DEBRIS: ["#hazard", "#debris"],
    T.POLICE_ACTIVITY: ["#police", "#cordon"], T.NONE: ["#London", "#weekend", "#sunshine"],
}  # fmt: skip

# --------------------------------------------------------------------------- label distributions

INCIDENT_WEIGHTS_LIVE = {T.NONE: 0.25, T.CONGESTION: 0.20, T.ACCIDENT: 0.10, T.BREAKDOWN: 0.10,
                         T.ROAD_WORKS: 0.10, T.WEATHER_HAZARD: 0.08, T.DEBRIS: 0.08,
                         T.POLICE_ACTIVITY: 0.09}  # fmt: skip
# Flatter for training data so rare classes get enough examples.
INCIDENT_WEIGHTS_DATASET = {t: (0.16 if t is T.NONE else 0.12) for t in IncidentType}

SEVERITY_WEIGHTS = {
    T.ACCIDENT: (0.15, 0.30, 0.35, 0.20), T.BREAKDOWN: (0.30, 0.40, 0.20, 0.10),
    T.ROAD_WORKS: (0.30, 0.40, 0.22, 0.08), T.CONGESTION: (0.25, 0.35, 0.28, 0.12),
    T.WEATHER_HAZARD: (0.25, 0.35, 0.28, 0.12), T.DEBRIS: (0.30, 0.40, 0.22, 0.08),
    T.POLICE_ACTIVITY: (0.30, 0.35, 0.25, 0.10),
}  # fmt: skip
STYLE_WEIGHTS = {
    Style.DRIVER: 0.35,
    Style.SOCIAL_MEDIA: 0.35,
    Style.OPERATOR: 0.15,
    Style.POLICE: 0.15,
}

# --------------------------------------------------------------------------- noise

ABBREVIATIONS = {"minutes": "mins", "because": "cos", "people": "ppl", "please": "pls", "road": "rd",
                 "about": "abt", "with": "w/", "right now": "rn", "near": "nr", "you": "u",
                 "traffic": "trafic", "before": "b4", "tonight": "2nite", "really": "rly"}  # fmt: skip
_KEYBOARD_NEIGHBOURS = {"a": "sq", "e": "wr", "i": "uo", "o": "ip", "n": "bm", "r": "et", "s": "ad",
                        "t": "ry", "l": "k", "c": "xv"}  # fmt: skip
_LOC = "\x00"  # sentinel marking where the location mention goes


@dataclass(frozen=True)
class NoiseProfile:
    lowercase: float = 0.0
    strip_punctuation: float = 0.0
    typo_rate: float = 0.0
    abbreviate: float = 0.0
    shout_word: float = 0.0


NOISE = {
    Style.OPERATOR: NoiseProfile(),
    Style.POLICE: NoiseProfile(typo_rate=0.005),
    Style.DRIVER: NoiseProfile(lowercase=0.6, strip_punctuation=0.4, typo_rate=0.07, abbreviate=0.5),
    Style.SOCIAL_MEDIA: NoiseProfile(lowercase=0.35, strip_punctuation=0.2, typo_rate=0.04,
                                     abbreviate=0.6, shout_word=0.3),
}  # fmt: skip


def _typo(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)
    kind = rng.random()
    if kind < 0.3:  # swap neighbours
        return word[: i - 1] + word[i] + word[i - 1] + word[i + 1 :]
    if kind < 0.55:  # drop
        return word[:i] + word[i + 1 :]
    if kind < 0.8:  # double
        return word[:i] + word[i] + word[i:]
    near = _KEYBOARD_NEIGHBOURS.get(word[i].lower())
    return word[:i] + rng.choice(near) + word[i + 1 :] if near else word


def _noisy(text: str, profile: NoiseProfile, rng: random.Random, abbreviate: bool) -> str:
    if abbreviate:
        for full, short in ABBREVIATIONS.items():
            if rng.random() < 0.5:
                # (?<!#): leave hashtags such as #traffic alone.
                text = re.sub(rf"(?<![#\w]){re.escape(full)}\b", short, text)
    if profile.typo_rate:
        words = text.split(" ")
        for j, w in enumerate(words):
            if len(w) > 3 and w.isalpha() and rng.random() < profile.typo_rate:
                words[j] = _typo(w, rng)
        text = " ".join(words)
    if profile.shout_word and rng.random() < profile.shout_word:
        words = text.split(" ")
        candidates = [j for j, w in enumerate(words) if len(w) > 3 and w.isalpha()]
        if candidates:
            j = rng.choice(candidates)
            words[j] = words[j].upper()
        text = " ".join(words)
    return text


@dataclass(frozen=True)
class GeneratedReport:
    text: str
    incident_type: IncidentType
    severity: Severity
    style: Style
    camera_id: str
    location_mention: str  # exact substring of text, or "" if the report names no place
    stage: Stage
    core: str = ""  # un-noised building blocks, for tests and debugging
    detail: str = ""
    text_style: str | None = None  # "slang" while drift mode is on; None = the standard style

    @property
    def source(self) -> ReportSource:
        return STYLE_SOURCE[self.style]


class ReportGenerator:
    def __init__(
        self,
        rng: random.Random | None = None,
        variant: Variant = Variant.LIVE,
        text_style: Callable[[], str | None] | None = None,
    ) -> None:
        self.rng = rng or random.Random()
        self.variant = variant
        # Asked for every report: drift mode switches the style to "slang" while it is on.
        self.text_style = text_style

    # -------------------------------------------------------------- label sampling

    def sample_incident_type(
        self, weights: dict[IncidentType, float] | None = None
    ) -> IncidentType:
        weights = weights or INCIDENT_WEIGHTS_LIVE
        return self.rng.choices(list(weights), weights=list(weights.values()))[0]

    def sample_severity(self, incident_type: IncidentType) -> Severity:
        if incident_type is T.NONE:
            return S.LOW
        return self.rng.choices(list(Severity), weights=SEVERITY_WEIGHTS[incident_type])[0]

    def sample_style(self, incident_type: IncidentType) -> Style:
        weights = dict(STYLE_WEIGHTS)
        if incident_type is T.POLICE_ACTIVITY:
            weights[Style.POLICE] *= 2.5
        if incident_type is T.ROAD_WORKS:
            weights[Style.OPERATOR] *= 2
        return self.rng.choices(list(weights), weights=list(weights.values()))[0]

    # -------------------------------------------------------------- generation

    def generate(
        self,
        camera: CameraInfo,
        *,
        incident_type: IncidentType | None = None,
        severity: Severity | None = None,
        style: Style | None = None,
        stage: Stage = Stage.ACTIVE,
        now: datetime | None = None,
        noise: bool = True,
        text_style: str | None = None,
    ) -> GeneratedReport:
        rng, variant = self.rng, self.variant
        incident_type = incident_type or self.sample_incident_type()
        if incident_type is T.NONE:
            stage = Stage.ACTIVE
        severity = S.LOW if stage is Stage.CLEARED or incident_type is T.NONE else (
            severity or self.sample_severity(incident_type))  # fmt: skip
        gaz = gazetteer_for(camera)
        text_style = text_style or (self.text_style() if self.text_style else None)
        if text_style == slang.STYLE:
            return self._generate_slang(camera, gaz, incident_type, severity, stage)
        style = style or self.sample_style(incident_type)

        if incident_type is T.NONE:
            template = CHATTER_TEMPLATES[style].pick(rng, variant)
        elif stage is Stage.CLEARED:
            template = CLEARED_TEMPLATES[style].pick(rng, variant)
        else:
            template = ACTIVE_TEMPLATES[style].pick(rng, variant)

        core = CORE[incident_type].pick(rng, variant) if incident_type is not T.NONE else ""
        detail = (
            DETAIL[incident_type][severity].pick(rng, variant)
            if incident_type is not T.NONE and stage is Stage.ACTIVE else ""
        )  # fmt: skip
        cleared = CLEARED_DETAIL.pick(rng, variant)
        prep, mention = self._location(gaz, style)
        now = now or datetime.now()
        values = {
            "core": core, "Core": core[:1].upper() + core[1:],
            "detail": detail, "Detail": detail[:1].upper() + detail[1:],
            "cleared": cleared, "Cleared": cleared[:1].upper() + cleared[1:],
            "loc": f"{prep} {_LOC}", "loc_bare": _LOC,
            "opener": OPENERS.pick(rng, variant),
            "closer": (POLICE_CLOSERS if style is Style.POLICE else OPERATOR_CLOSERS).pick(rng, variant),
            "emoji": rng.choice(TYPE_EMOJI[incident_type]),
            "tags": self._tags(incident_type, gaz),
            "time": now.strftime("%H:%M"),
            "ref": f"INC-{rng.randint(1000, 99999)}",
        }  # fmt: skip
        raw = template.format(**values)
        text, mention_out = self._finish(raw, mention, style, noise)
        return GeneratedReport(
            text=text, incident_type=incident_type, severity=severity, style=style,
            camera_id=camera.id, location_mention=mention_out, stage=stage, core=core, detail=detail,
        )  # fmt: skip

    def _generate_slang(
        self,
        camera: CameraInfo,
        gaz: Gazetteer,
        incident_type: IncidentType,
        severity: Severity,
        stage: Stage,
    ) -> GeneratedReport:
        """Drift mode's report style: posted by drivers and on social media, in slang."""
        rng = self.rng
        style = rng.choice([Style.DRIVER, Style.SOCIAL_MEDIA])
        template, core = slang.compose(rng, incident_type, severity, stage is Stage.CLEARED)
        _, mention = self._location(gaz, style)
        raw = template.replace("{loc}", f"{rng.choice(slang.PREPOSITIONS)} {_LOC}", 1)
        before, _, after = raw.partition(_LOC)
        before, after = slang.text_speak(before, rng), slang.text_speak(after, rng)
        if rng.random() < 0.7:
            before, after, mention = before.lower(), after.lower(), mention.lower()
        text = re.sub(r"\s+", " ", before + mention + after).strip()
        return GeneratedReport(
            text=text, incident_type=incident_type, severity=severity, style=style,
            camera_id=camera.id, location_mention=mention, stage=stage, core=core,
            text_style=slang.STYLE,
        )  # fmt: skip

    def _location(self, gaz: Gazetteer, style: Style) -> tuple[str, str]:
        rng = self.rng
        kinds = ["official", "street", "landmark", "abbreviation", "misspelling", "compound"]
        weights = [0.22, 0.25, 0.15, 0.13, 0.1, 0.15]
        if style in (Style.OPERATOR, Style.POLICE):
            weights[4] = 0.0  # official feeds spell place names correctly
        if self.variant is not Variant.TRAIN and gaz.holdout:
            kinds.append("holdout")
            weights.append(0.3 if self.variant is Variant.TEST else 0.05)
        kind = rng.choices(kinds, weights=weights)[0]
        if kind == "street":
            return rng.choice(["on", "along", "on", "at"]), rng.choice(gaz.streets)
        if kind == "landmark":
            return rng.choice(["near", "by", "outside", "next to"]), rng.choice(gaz.landmarks)
        if kind == "compound":
            return "on", f"{rng.choice(gaz.streets)} near {rng.choice(gaz.landmarks)}"
        pool = {"official": gaz.official, "abbreviation": gaz.abbreviations,
                "misspelling": gaz.misspellings, "holdout": gaz.holdout}[kind]  # fmt: skip
        return rng.choice(["at", "near", "around", "approaching"]), rng.choice(pool)

    def _tags(self, incident_type: IncidentType, gaz: Gazetteer) -> str:
        options = ["#LondonTraffic", f"#{gaz.hashtag}", *TYPE_TAGS[incident_type]]
        return " ".join(self.rng.sample(options, k=self.rng.randint(1, min(3, len(options)))))

    def _finish(self, raw: str, mention: str, style: Style, noise: bool) -> tuple[str, str]:
        """Apply noise around (never inside) the location mention; return text and final mention."""
        has_location = _LOC in raw
        before, _, after = raw.partition(_LOC) if has_location else (raw, "", "")
        if noise:
            profile = NOISE[style]
            abbreviate = self.rng.random() < profile.abbreviate
            before = _noisy(before, profile, self.rng, abbreviate)
            after = _noisy(after, profile, self.rng, abbreviate)
            if self.rng.random() < profile.lowercase:
                before, after, mention = before.lower(), after.lower(), mention.lower()
            if self.rng.random() < profile.strip_punctuation:
                before = re.sub(r"[.,!?;:]", "", before)
                after = re.sub(r"[.,!?;:]", "", after)
        text = before + (mention if has_location else "") + after
        text = re.sub(r"\s+", " ", text).strip()
        mention = mention if has_location else ""
        return text, mention
