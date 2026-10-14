"""Traffic and weather models behind the structured events."""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime

# --------------------------------------------------------------------------- traffic


def _bump(hour: float, centre: float, width: float) -> float:
    return math.exp(-0.5 * ((hour - centre) / width) ** 2)


def daily_flow_fraction(local_time: datetime) -> float:
    """Expected flow as a fraction of capacity (0..1) at a local time.

    Weekdays: low overnight, a sharp morning peak around 08:00, a midday plateau and a
    broader evening peak around 17:30. Weekends: no commuter peaks, a late-morning to
    afternoon hump instead.
    """
    hour = local_time.hour + local_time.minute / 60 + local_time.second / 3600
    night = 0.06
    if local_time.weekday() >= 5:
        return min(1.0, night + 0.55 * _bump(hour, 13.5, 3.5))
    return min(
        1.0,
        night
        + 0.80 * _bump(hour, 8.0, 1.1)
        + 0.40 * _bump(hour, 12.5, 2.5)
        + 0.85 * _bump(hour, 17.5, 1.5),
    )


@dataclass(frozen=True)
class RoadProfile:
    capacity_veh_per_hour: float = 1800.0
    free_flow_kmh: float = 48.0  # London 30 mph roads
    min_speed_kmh: float = 4.0


def speed_for_flow(fraction: float, road: RoadProfile) -> float:
    """Speed falls off as flow approaches capacity (a simple BPR-style curve)."""
    slowdown = 0.15 * (fraction / 0.25) ** 0.5 if fraction < 0.25 else 0.15 + 0.75 * (
        (fraction - 0.25) / 0.75) ** 2  # fmt: skip
    return max(road.min_speed_kmh, road.free_flow_kmh * (1 - min(slowdown, 0.92)))


@dataclass
class SensorReading:
    vehicle_count: int
    flow_veh_per_hour: float
    avg_speed_kmh: float
    occupancy: float
    interval_seconds: float


def sensor_reading(
    local_time: datetime,
    interval_seconds: float,
    rng: random.Random,
    road: RoadProfile | None = None,
    count_factor: float = 1.0,
    speed_factor: float = 1.0,
) -> SensorReading:
    """One loop-detector reading. Factors come from scenarios (e.g. speed_factor 0.2 = crawl)."""
    road = road or RoadProfile()
    fraction = daily_flow_fraction(local_time)
    expected_count = road.capacity_veh_per_hour * fraction * count_factor * interval_seconds / 3600
    # Poisson-like noise (normal approximation, sd = sqrt(mean)).
    count = max(0, round(rng.gauss(expected_count, math.sqrt(max(expected_count, 0.5)))))
    speed = speed_for_flow(fraction, road) * speed_factor * rng.uniform(0.92, 1.08)
    speed = max(road.min_speed_kmh * min(speed_factor, 1.0), speed)
    flow = count * 3600 / interval_seconds
    occupancy = min(0.98, fraction * count_factor * (road.free_flow_kmh / max(speed, 1.0)) * 0.35)
    return SensorReading(
        count, round(flow, 1), round(speed, 1), round(occupancy, 3), interval_seconds
    )


# --------------------------------------------------------------------------- weather

# Typical values per condition: (visibility_m, precipitation_mm_h)
CONDITIONS = {
    "clear": (10_000, 0.0),
    "cloudy": (8_000, 0.0),
    "light_rain": (5_000, 1.5),
    "rain": (3_000, 5.0),
    "heavy_rain": (1_200, 18.0),
    "fog": (150, 0.0),
    "snow": (1_000, 2.0),
}
_TRANSITIONS = {
    "clear": {"cloudy": 0.8, "fog": 0.2},
    "cloudy": {"clear": 0.45, "light_rain": 0.45, "fog": 0.1},
    "light_rain": {"cloudy": 0.5, "rain": 0.5},
    "rain": {"light_rain": 0.6, "heavy_rain": 0.4},
    "heavy_rain": {"rain": 1.0},
    "fog": {"cloudy": 0.6, "clear": 0.4},
    "snow": {"cloudy": 1.0},
}


@dataclass
class WeatherState:
    condition: str = "cloudy"
    temperature_c: float = 12.0
    visibility_m: float = 8_000.0
    precipitation_mm_h: float = 0.0


@dataclass
class WeatherModel:
    """Gradual weather: conditions change rarely (Markov chain), values move smoothly
    toward the condition's typical levels, temperature follows a daily cycle.

    ``override`` (set by scenarios) pins the target condition/values until cleared.
    """

    rng: random.Random
    state: WeatherState = field(default_factory=WeatherState)
    change_probability: float = 0.015  # per tick; ~15 min mean at a 15 s tick
    smoothing: float = 0.12  # fraction of the gap to the target closed per tick
    base_temperature_c: float = 12.0
    override: dict[str, float | str] | None = None

    def tick(self, local_time: datetime) -> WeatherState:
        s = self.state
        if self.override and "condition" in self.override:
            s.condition = str(self.override["condition"])
        elif self.rng.random() < self.change_probability:
            options = _TRANSITIONS[s.condition]
            s.condition = self.rng.choices(list(options), weights=list(options.values()))[0]

        visibility, precipitation = CONDITIONS[s.condition]
        if self.override:
            visibility = float(self.override.get("visibility_m", visibility))
            precipitation = float(self.override.get("precipitation_mm_h", precipitation))
        hour = local_time.hour + local_time.minute / 60
        temperature = (
            self.base_temperature_c
            + 4.0 * math.sin((hour - 9) / 24 * 2 * math.pi)  # warmest mid-afternoon
            - (2.0 if precipitation > 3 else 0.0)
        )
        if self.override and "temperature_c" in self.override:
            temperature = float(self.override["temperature_c"])

        k = self.smoothing
        s.visibility_m += k * (visibility - s.visibility_m) + self.rng.gauss(0, visibility * 0.01)
        s.precipitation_mm_h += k * (precipitation - s.precipitation_mm_h)
        s.temperature_c += k * (temperature - s.temperature_c) + self.rng.gauss(0, 0.05)
        s.visibility_m = max(20.0, s.visibility_m)
        s.precipitation_mm_h = max(0.0, s.precipitation_mm_h)
        return s

    def as_payload(self) -> dict[str, float | str]:
        s = self.state
        return {
            "condition": s.condition,
            "temperature_c": round(s.temperature_c, 1),
            "visibility_m": round(s.visibility_m),
            "precipitation_mm_h": round(s.precipitation_mm_h, 1),
        }
