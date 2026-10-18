"""The "slang" report style that drift mode switches to (traffic_common.driftmode).

New slang, abbreviations and phrasing that never appear in the standard phrase pools, so a
classifier trained on the standard style sees genuinely unfamiliar text. The labels stay
recoverable from the text: each incident type has its own slang phrases, and each severity
its own impact phrases, so a model retrained on labelled slang reports can learn them.
"""

import random
import re

from traffic_common.enums import IncidentType, Severity

T = IncidentType
S = Severity
STYLE = "slang"

CORE: dict[IncidentType, tuple[str, ...]] = {
    T.ACCIDENT: ("a proper smash-up", "two motors went bump", "a nasty prang", "a fender bender",
                 "someone's totalled their whip", "a bike vs car situation", "a pile-up"),
    T.BREAKDOWN: ("some geezer's motor has died", "a van that's given up the ghost",
                  "a dead lorry", "a car that's packed in", "a bus that's croaked",
                  "a motor with smoke coming out the bonnet"),
    T.ROAD_WORKS: ("the council digging the road up AGAIN", "diggers everywhere",
                   "workmen with cones out", "a hole in the road w/ a crew round it",
                   "the gas board ripping up the tarmac", "hi-vis lads blocking a lane"),
    T.CONGESTION: ("a car park where the road should be", "zero movement", "mad traffic",
                   "the usual crawl", "a sea of brake lights", "traffic going nowhere"),
    T.WEATHER_HAZARD: ("the road turned into a river", "a proper pea-souper", "an ice rink of a road",
                       "puddles the size of lakes", "a tree that's keeled over", "slippy af tarmac"),
    T.DEBRIS: ("random junk all over the road", "someone's load gone flying", "bits of a car everywhere",
               "a sofa just chilling in the lane", "rubble all over", "a ladder in the middle of the road"),
    T.POLICE_ACTIVITY: ("the feds everywhere", "old bill blocking it off", "blue lights all over",
                        "cops shutting the road", "a load of police vans", "fuzz waving ppl round"),
}  # fmt: skip

IMPACT: dict[Severity, tuple[str, ...]] = {
    S.LOW: ("nbd tho, still moving", "no biggie, everyone's fine", "barely a delay tbh",
            "all good fam, just mind it"),
    S.MEDIUM: ("one lane's gone so it's slow", "bit of a wait ngl", "kinda backed up",
               "lane shut-ish, crawling past"),
    S.HIGH: ("two lanes dead, massive wait", "proper messy, ambos there", "huge queue, add an hour",
             "absolute nightmare, barely moving"),
    S.CRITICAL: ("whole road's shut, nothing getting thru", "absolute carnage, avoid avoid avoid",
                 "air ambo landing, road closed", "total shutdown, turn round now"),
}  # fmt: skip

CLEARED = ("all sorted now", "back to normal innit", "moving again finally", "road's open again")
CHATTER = (
    "vibes are immaculate {loc} today", "who's got the best brekkie {loc}?? asking for a mate",
    "lowkey love the sunset {loc}", "my playlist is slapping on this drive {loc}",
    "nothing to report {loc}, smooth sailing", "just vibing {loc} ngl",
)  # fmt: skip

TEMPLATES = (
    "yo {core} {loc} rn, {impact} smh", "ngl {core} {loc}... {impact} 💀",
    "PSA: {core} {loc} fr fr. {impact}", "lowkey {core} {loc}, {impact} lol",
    "{core} {loc} innit, {impact}", "bruh. {core} {loc}. {impact} 😤",
    "heads up fam - {core} {loc}, {impact}", "can't make this up: {core} {loc}, {impact} 🙃",
)  # fmt: skip
CLEARED_TEMPLATES = ("{core} {loc} is done, {cleared} 🙌", "update: {core} {loc} gone, {cleared}")
PREPOSITIONS = ("@", "by", "near", "outside")

# Text-speak applied on top (only outside the location mention).
ABBREVIATIONS = {"to": "2", "for": "4", "you": "u", "your": "ur", "because": "cuz",
                 "people": "ppl", "about": "abt", "really": "rly", "though": "tho",
                 "through": "thru", "probably": "prob", "everyone": "every1", "and": "n"}  # fmt: skip


def text_speak(text: str, rng: random.Random, rate: float = 0.6) -> str:
    for full, short in ABBREVIATIONS.items():
        if rng.random() < rate:
            text = re.sub(rf"(?<![#\w]){re.escape(full)}\b", short, text)
    return text


def compose(
    rng: random.Random,
    incident_type: IncidentType,
    severity: Severity,
    cleared: bool,
) -> tuple[str, str]:
    """A slang template with {loc} left in place, and the core phrase used."""
    if incident_type is T.NONE:
        return rng.choice(CHATTER), ""
    core = rng.choice(CORE[incident_type])
    if cleared:
        template = rng.choice(CLEARED_TEMPLATES)
        return template.format(core=core, loc="{loc}", cleared=rng.choice(CLEARED)), core
    template = rng.choice(TEMPLATES)
    return template.format(core=core, loc="{loc}", impact=rng.choice(IMPACT[severity])), core
