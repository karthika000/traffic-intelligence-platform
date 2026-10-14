"""Scenario definitions (YAML) and their timelines.

A scenario is a sequence of phases. Each phase lasts ``duration`` (e.g. ``90s``, ``4m``)
and sets: the severity and incident type(s) of the reports it emits, how often and from
which sources, the report stage (``active`` or ``cleared``), the traffic it causes
(speed and count factors relative to the normal daily pattern, ramped in over
``ramp``), and optionally a weather override. See scenarios/*.yaml.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from event_simulator.patterns import CONDITIONS
from event_simulator.reports import Stage, Style
from traffic_common.enums import IncidentType, Severity

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|m|h)?\s*$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, None: 1}
_SEVERITY_ORDER = list(Severity)


def parse_duration(value: Any) -> float:
    """``90``, ``"90s"``, ``"4m"``, ``"1.5h"`` -> seconds."""
    if isinstance(value, int | float):
        return float(value)
    match = _DURATION_RE.match(str(value))
    if not match:
        raise ValueError(f"invalid duration {value!r}; use e.g. 90s, 4m, 1h")
    return float(match.group(1)) * _UNIT_SECONDS[match.group(2)]


Seconds = Annotated[float, BeforeValidator(parse_duration), Field(gt=0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReportsSpec(_Strict):
    per_minute: float = Field(ge=0)
    sources: dict[Style, float] = Field(min_length=1)  # style -> relative weight
    # Defaults to the scenario's incident_type; several types allow e.g. congestion side reports.
    incident_types: dict[IncidentType, float] | None = None
    stage: Stage = Stage.ACTIVE


class TrafficSpec(_Strict):
    speed_factor: float = Field(default=1.0, gt=0, le=2)
    count_factor: float = Field(default=1.0, gt=0, le=3)


class WeatherSpec(_Strict):
    condition: str
    visibility_m: float | None = Field(default=None, gt=0)
    precipitation_mm_h: float | None = Field(default=None, ge=0)
    temperature_c: float | None = None

    @model_validator(mode="after")
    def _known_condition(self) -> "WeatherSpec":
        if self.condition not in CONDITIONS:
            raise ValueError(
                f"unknown weather condition {self.condition!r}; use {list(CONDITIONS)}"
            )
        return self

    def as_override(self) -> dict[str, float | str]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class Phase(_Strict):
    name: str = Field(min_length=1, max_length=50)
    duration: Seconds
    severity: Severity
    reports: ReportsSpec | None = None
    traffic: TrafficSpec = TrafficSpec()
    ramp: Seconds = 60.0  # time to move traffic factors from the previous phase's values
    weather: WeatherSpec | None = None


class Scenario(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    incident_type: IncidentType
    phases: list[Phase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_phase_names(self) -> "Scenario":
        names = [p.name for p in self.phases]
        if len(set(names)) != len(names):
            raise ValueError("phase names must be unique")
        return self

    @property
    def peak_severity(self) -> Severity:
        return max((p.severity for p in self.phases), key=_SEVERITY_ORDER.index)

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.phases)

    def timeline(self, time_scale: float = 1.0) -> list["PhaseWindow"]:
        """Phases with start/end offsets in (scaled) seconds from the start of the run."""
        windows, start = [], 0.0
        previous = TrafficSpec()
        for phase in self.phases:
            end = start + phase.duration / time_scale
            windows.append(PhaseWindow(phase, start, end, previous, time_scale))
            previous, start = phase.traffic, end
        return windows

    def report_incident_types(self, phase: Phase) -> dict[IncidentType, float]:
        if phase.reports and phase.reports.incident_types:
            return dict(phase.reports.incident_types)
        return {self.incident_type: 1.0}


@dataclass(frozen=True)
class PhaseWindow:
    phase: Phase
    start: float
    end: float
    previous_traffic: TrafficSpec
    time_scale: float

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    def traffic_factors(self, t: float) -> tuple[float, float]:
        """(speed_factor, count_factor) at offset ``t``, ramped linearly from the previous phase."""
        ramp = self.phase.ramp / self.time_scale
        progress = 1.0 if ramp <= 0 else min(1.0, max(0.0, (t - self.start) / ramp))
        prev, cur = self.previous_traffic, self.phase.traffic
        return (
            prev.speed_factor + (cur.speed_factor - prev.speed_factor) * progress,
            prev.count_factor + (cur.count_factor - prev.count_factor) * progress,
        )

    def as_record(self, started_at_iso: str | None = None) -> dict[str, Any]:
        return {
            "phase": self.phase.name,
            "severity": self.phase.severity.value,
            "start_offset_s": round(self.start, 3),
            "end_offset_s": round(self.end, 3),
            "stage": (self.phase.reports.stage if self.phase.reports else Stage.ACTIVE).value,
            "speed_factor": self.phase.traffic.speed_factor,
            "count_factor": self.phase.traffic.count_factor,
            "weather": self.phase.weather.as_override() if self.phase.weather else None,
        }


def load_scenario(path: Path) -> Scenario:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a mapping at the top level")
    scenario = Scenario.model_validate(data)
    if scenario.name != path.stem:
        raise ValueError(f"{path.name}: name {scenario.name!r} must match the file name")
    return scenario


def load_scenarios(directory: Path) -> dict[str, Scenario]:
    return {p.stem: load_scenario(p) for p in sorted(directory.glob("*.yaml"))}
