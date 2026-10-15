"""Per-camera zone configuration (cameras.zone_config JSON).

    {
      "roadway": [[0.0, 0.41], [1.0, 0.44], [1.0, 0.97], [0.0, 0.97]],   # normalised x, y
      "pixels_per_metre": 46.0,          # horizontal image scale
      "pixels_per_metre_y": 12.0,        # optional vertical scale (depth is foreshortened)
      "reference_width": 854             # frame width the scales were measured at
    }

Polygon points are normalised to [0, 1] so they survive resolution changes; the scales
are rescaled from ``reference_width`` to the actual frame width. Missing roadway -> the
whole frame; missing scale -> ``frame width / default_frame_width_metres``. Unknown keys
(e.g. older configs) are ignored. A single scale is a deliberate simplification: speeds
are most accurate where vehicles cross the area the scale was measured in.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Point = tuple[float, float]


class ZoneConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    roadway: list[tuple[float, float]] | None = Field(default=None, min_length=3)
    pixels_per_metre: float | None = Field(default=None, gt=0)
    pixels_per_metre_y: float | None = Field(default=None, gt=0)
    reference_width: int | None = Field(default=None, gt=0)

    @field_validator("roadway")
    @classmethod
    def _normalised(
        cls, points: list[tuple[float, float]] | None
    ) -> list[tuple[float, float]] | None:
        if points is not None:
            for x, y in points:
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("roadway points must be normalised to [0, 1]")
        return points

    @classmethod
    def parse(cls, raw: dict[str, Any] | None) -> tuple["ZoneConfig", str | None]:
        """Lenient parse for configs read from the database: returns (config, problem)."""
        try:
            return cls.model_validate(raw or {}), None
        except ValidationError as exc:
            return cls(), f"invalid zone_config, using defaults: {exc.errors()[0]['msg']}"

    def resolve(self, width: int, height: int, default_frame_width_metres: float) -> "ResolvedZone":
        if self.roadway:
            polygon = [(x * width, y * height) for x, y in self.roadway]
        else:
            polygon = [
                (0.0, 0.0),
                (float(width), 0.0),
                (float(width), float(height)),
                (0.0, float(height)),
            ]
        if self.pixels_per_metre:
            scale = width / self.reference_width if self.reference_width else 1.0
            ppm_x = self.pixels_per_metre * scale
            ppm_y = (self.pixels_per_metre_y or self.pixels_per_metre) * scale
        else:
            ppm_x = ppm_y = width / default_frame_width_metres
        return ResolvedZone(
            polygon,
            ppm_x,
            ppm_y,
            whole_frame=not self.roadway,
            default_scale=not self.pixels_per_metre,
        )


@dataclass(frozen=True)
class ResolvedZone:
    """A zone in pixel coordinates for one frame size."""

    polygon: list[Point]
    ppm_x: float
    ppm_y: float
    whole_frame: bool = False
    default_scale: bool = False

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon (ray casting). Points on an edge may fall either way."""
        inside = False
        pts = self.polygon
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside

    def to_metres(self, x: float, y: float) -> Point:
        return x / self.ppm_x, y / self.ppm_y


def ground_point(bbox: tuple[float, float, float, float]) -> Point:
    """Where an object touches the road: bottom centre of its box (pixels)."""
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) / 2, y2
