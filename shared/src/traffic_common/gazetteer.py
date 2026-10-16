"""How people refer to each camera's location: official names, streets, landmarks,
abbreviations and misspellings. Demo cameras have a hand-written gazetteer; other
cameras get one derived from their name and location.

Shared by the event simulator (which writes reports using these names) and nlp_service
(which matches reports back to cameras with them). ``holdout`` forms are only used by the
simulator's test split; the matcher must not read them, or its evaluation would be leaked."""

import random
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraInfo:
    id: str
    name: str
    location_name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Gazetteer:
    official: list[str]  # e.g. "Hyde Park Corner"
    streets: list[str]
    landmarks: list[str]
    abbreviations: list[str]
    misspellings: list[str]
    hashtag: str  # area hashtag for social media posts
    # Only used for the test split, so the model is evaluated on unseen location forms.
    holdout: list[str] = field(default_factory=list)


GAZETTEERS: dict[str, Gazetteer] = {
    "cam-hyde-park-corner": Gazetteer(
        official=["Hyde Park Corner", "Hyde Park Corner roundabout"],
        streets=[
            "Piccadilly",
            "Knightsbridge",
            "Grosvenor Place",
            "Park Lane",
            "Constitution Hill",
            "Grosvenor Crescent",
        ],
        landmarks=[
            "Wellington Arch",
            "Apsley House",
            "the Hyde Park Corner underpass",
            "the Lanesborough",
        ],
        abbreviations=["HPC", "Hyde Pk Corner", "Hyde Pk Cnr", "Park Ln"],
        misspellings=["Hide Park Corner", "Hyde Prk Corner", "hydepark corner", "Picadilly"],
        hashtag="HydeParkCorner",
        holdout=["Duke of Wellington Place", "the Apsley Gate", "Hyde Park Crnr"],
    ),
    "cam-trafalgar-square": Gazetteer(
        official=["Trafalgar Square", "Trafalgar Square gyratory"],
        streets=[
            "the Strand",
            "Whitehall",
            "Charing Cross Road",
            "Pall Mall East",
            "Cockspur Street",
            "Northumberland Avenue",
        ],
        landmarks=[
            "Nelson's Column",
            "the National Gallery",
            "Admiralty Arch",
            "Charing Cross station",
        ],
        abbreviations=["Trafalgar Sq", "Traf Sq", "Charing X", "Northumberland Ave"],
        misspellings=["Trafalger Square", "Trafalgar Sqaure", "Nelsons collumn", "Whitehal"],
        hashtag="TrafalgarSquare",
        holdout=["Duncannon Street", "St Martin-in-the-Fields", "Trafalgr Sq"],
    ),
    "cam-elephant-castle": Gazetteer(
        official=["Elephant & Castle", "Elephant and Castle northern roundabout"],
        streets=[
            "Newington Causeway",
            "London Road",
            "St George's Road",
            "Walworth Road",
            "New Kent Road",
            "Newington Butts",
        ],
        landmarks=[
            "Elephant & Castle station",
            "the Faraday Memorial",
            "Elephant Park",
            "the old shopping centre site",
        ],
        abbreviations=["E&C", "Elephant", "Ele & Castle", "New Kent Rd"],
        misspellings=["Elephant and Castel", "Elefant & Castle", "Newington Cswy", "Walworth Rd"],
        hashtag="ElephantAndCastle",
        holdout=["Brook Drive", "the Metropolitan Tabernacle", "Elephnt & Castle"],
    ),
    "cam-old-street": Gazetteer(
        official=["Old Street roundabout", "Old Street"],
        streets=[
            "City Road",
            "Great Eastern Street",
            "East Road",
            "Clerkenwell Road",
            "Old Street westbound",
            "Pitfield Street",
        ],
        landmarks=[
            "Silicon Roundabout",
            "Old Street station",
            "Moorfields Eye Hospital",
            "Bunhill Fields",
        ],
        abbreviations=["Old St", "Old St rbt", "City Rd", "Gt Eastern St"],
        misspellings=["Olde Street", "Old Steet", "silicon roundabot", "Moorfeilds"],
        hashtag="OldStreet",
        holdout=["Featherstone Street", "the Old Street tube exit", "Old Strt"],
    ),
}

_STREET_SUFFIXES = {"Street": "St", "Road": "Rd", "Avenue": "Ave", "Square": "Sq", "Lane": "Ln"}


def _misspell(word: str, rng: random.Random) -> str:
    letters = [i for i, ch in enumerate(word) if ch.isalpha()]
    if len(letters) < 4:
        return word
    i = rng.choice(letters[1:-1])
    return word[:i] + word[i + 1 :]  # drop one letter


def derived_gazetteer(camera: CameraInfo) -> Gazetteer:
    """A plausible gazetteer for a camera without a hand-written one."""
    rng = random.Random(camera.id)
    place = camera.location_name.split(",")[0].strip() or camera.name
    parts = [p.strip() for p in re.split(r"\s+-\s+|,", camera.name) if p.strip()]
    abbreviated = place
    for full, short in _STREET_SUFFIXES.items():
        abbreviated = re.sub(rf"\b{full}\b", short, abbreviated)
    hashtag = re.sub(r"[^A-Za-z0-9]", "", place.title()) or "Traffic"
    return Gazetteer(
        official=[place],
        streets=parts[1:] or [place],
        landmarks=[f"the {place} junction"],
        abbreviations=[abbreviated] if abbreviated != place else [place],
        misspellings=[_misspell(place, rng)],
        hashtag=hashtag,
    )


def gazetteer_for(camera: CameraInfo) -> Gazetteer:
    return GAZETTEERS.get(camera.id) or derived_gazetteer(camera)
