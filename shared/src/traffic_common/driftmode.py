"""Drift mode: a demo switch that degrades the platform's inputs so drift monitoring has
something to find.

- **Visual** (video_ingest): every frame is darkened, hazed, blurred and/or noised before it
  is encoded, to look like night or fog.
- **Text** (event_simulator): reports are written in a new style (slang, abbreviations and
  phrasing that never appear in the training data).

The state is one row in the ``drift_mode`` table, set through the API
(``POST /api/v1/drift/mode``) or ``python -m traffic_common.driftmode on|off``; every change
is also appended to ``drift_mode_events``. Services poll it with :class:`DriftModeWatcher`, so
switching it off takes effect within a few seconds. ``expires_at`` switches it off by itself.
"""

import argparse
import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, text

log = logging.getLogger(__name__)

VisualPreset = Literal["night", "fog", "custom"]

# brightness: multiplier (1 = unchanged); noise_std: Gaussian noise in grey levels;
# haze: blend towards a light grey (fog); blur: Gaussian kernel size in pixels (0 = none).
VISUAL_PRESETS: dict[str, dict[str, float]] = {
    "night": {"brightness": 0.3, "noise_std": 22.0, "haze": 0.0, "blur": 0.0},
    "fog": {"brightness": 0.9, "noise_std": 12.0, "haze": 0.65, "blur": 7.0},
}


class DriftModeSettings(BaseModel):
    """What drift mode changes. Unset visual parameters come from the preset."""

    model_config = ConfigDict(extra="forbid")

    visual: bool = True  # degrade frames in video_ingest
    text: bool = True  # switch event_simulator to the new report style
    preset: VisualPreset = "night"
    brightness: float | None = Field(default=None, ge=0.05, le=1.0)
    noise_std: float | None = Field(default=None, ge=0.0, le=80.0)
    haze: float | None = Field(default=None, ge=0.0, le=0.95)
    blur: float | None = Field(default=None, ge=0.0, le=25.0)
    text_style: Literal["slang"] = "slang"

    def visual_params(self) -> dict[str, float]:
        base = dict(VISUAL_PRESETS.get(self.preset, VISUAL_PRESETS["night"]))
        for key in base:
            value = getattr(self, key)
            if value is not None:
                base[key] = float(value)
        return base


class DriftModeState(BaseModel):
    enabled: bool = False
    settings: DriftModeSettings = Field(default_factory=DriftModeSettings)
    expires_at: datetime | None = None
    updated_by: str | None = None
    updated_at: datetime | None = None

    def active(self, now: datetime | None = None) -> bool:
        """Enabled and not expired."""
        now = now or datetime.now(UTC)
        return self.enabled and (self.expires_at is None or self.expires_at > now)

    def visual_active(self, now: datetime | None = None) -> bool:
        return self.active(now) and self.settings.visual

    def text_style(self, now: datetime | None = None) -> str | None:
        """The report style to write in, or None for the standard style."""
        return self.settings.text_style if self.active(now) and self.settings.text else None


SELECT_SQL = (
    "SELECT enabled, settings, expires_at, updated_by, updated_at FROM drift_mode WHERE id = 1"
)
UPSERT_SQL = """
INSERT INTO drift_mode (id, enabled, settings, expires_at, updated_by, updated_at)
VALUES (1, :enabled, CAST(:settings AS jsonb), :expires_at, :updated_by, now())
ON CONFLICT (id) DO UPDATE SET enabled = EXCLUDED.enabled, settings = EXCLUDED.settings,
    expires_at = EXCLUDED.expires_at, updated_by = EXCLUDED.updated_by, updated_at = now()
RETURNING enabled, settings, expires_at, updated_by, updated_at
"""
EVENT_SQL = """
INSERT INTO drift_mode_events (enabled, settings, expires_at, changed_by)
VALUES (:enabled, CAST(:settings AS jsonb), :expires_at, :updated_by)
"""
# Periods with drift mode on that overlap [:since, :until): reference windows and training
# data must avoid them. A period still running ends now.
ACTIVE_PERIODS_SQL = """
WITH ev AS (
    SELECT enabled, expires_at, changed_at,
           lead(changed_at) OVER (ORDER BY changed_at, id) AS next_change
    FROM drift_mode_events
)
SELECT changed_at AS start,
       LEAST(COALESCE(next_change, now()),
             COALESCE(expires_at, 'infinity'::timestamptz)) AS "end"
FROM ev WHERE enabled
  AND changed_at < :until
  AND LEAST(COALESCE(next_change, now()),
            COALESCE(expires_at, 'infinity'::timestamptz)) > :since
"""


def state_from_row(row: Any) -> DriftModeState:
    if row is None:
        return DriftModeState()
    m = row._mapping if hasattr(row, "_mapping") else row
    settings = m["settings"] or {}
    if isinstance(settings, str):
        settings = json.loads(settings)
    return DriftModeState(
        enabled=bool(m["enabled"]),
        settings=DriftModeSettings.model_validate(settings),
        expires_at=m["expires_at"],
        updated_by=m["updated_by"],
        updated_at=m["updated_at"],
    )


def write_params(
    enabled: bool,
    settings: DriftModeSettings | None,
    duration_minutes: float | None,
    updated_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    expires = now + timedelta(minutes=duration_minutes) if enabled and duration_minutes else None
    return {
        "enabled": enabled,
        "settings": (settings or DriftModeSettings()).model_dump_json(),
        "expires_at": expires,
        "updated_by": updated_by[:50],
    }


def read_state(engine: Engine) -> DriftModeState:
    with engine.connect() as conn:
        return state_from_row(conn.execute(text(SELECT_SQL)).first())


def set_state(
    engine: Engine,
    enabled: bool,
    settings: DriftModeSettings | None = None,
    *,
    duration_minutes: float | None = None,
    updated_by: str = "cli",
) -> DriftModeState:
    """Switch drift mode on or off (keeping the given settings) and record the change."""
    params = write_params(enabled, settings, duration_minutes, updated_by)
    with engine.begin() as conn:
        row = conn.execute(text(UPSERT_SQL), params).first()
        conn.execute(text(EVENT_SQL), params)
    return state_from_row(row)


def active_periods(
    engine: Engine, since: datetime, until: datetime
) -> list[tuple[datetime, datetime]]:
    with engine.connect() as conn:
        rows = conn.execute(text(ACTIVE_PERIODS_SQL), {"since": since, "until": until}).all()
    return [(r.start, r.end) for r in rows]


class DriftModeWatcher(threading.Thread):
    """Polls the drift_mode row every ``interval`` seconds and keeps the latest state.

    Read errors keep the last known state (logged once until it recovers); a service that
    never managed to read it sees drift mode off. ``on_change`` is called with each new state.
    """

    def __init__(
        self,
        engine: Engine | None,
        stop: threading.Event,
        interval: float = 5.0,
        on_change: Callable[[DriftModeState], None] | None = None,
        reader: Callable[[], DriftModeState] | None = None,
    ) -> None:
        super().__init__(name="drift-mode", daemon=True)
        self._reader = reader or (lambda: read_state(engine))  # type: ignore[arg-type]
        self._stop_event = stop
        self.interval = interval
        self.on_change = on_change
        self._state = DriftModeState()
        self._active: bool | None = None
        self._failing = False

    @property
    def state(self) -> DriftModeState:
        return self._state

    def poll(self) -> DriftModeState:
        try:
            state = self._reader()
        except Exception as exc:
            if not self._failing:
                log.warning("could not read drift mode (keeping the last state): %s", exc)
                self._failing = True
            return self._state
        self._failing = False
        active = state.active()
        if active != self._active or (active and state != self._state):
            if active:
                log.warning(
                    "DRIFT MODE ON (by %s): visual=%s %s, text=%s%s",
                    state.updated_by,
                    state.settings.visual,
                    state.settings.visual_params(),
                    state.text_style(),
                    f", until {state.expires_at:%H:%M:%S} UTC" if state.expires_at else "",
                )
            elif self._active is not None:
                log.warning("drift mode off")
            self._active = active
            if self.on_change:
                self.on_change(state)
        self._state = state
        return state

    def run(self) -> None:
        while True:
            self.poll()
            if self._stop_event.wait(self.interval):
                return


def main() -> None:
    from traffic_common.config import get_settings
    from traffic_common.db import create_db_engine

    parser = argparse.ArgumentParser(description="Show or switch the demo drift mode.")
    parser.add_argument("action", choices=["status", "on", "off"], nargs="?", default="status")
    parser.add_argument("--preset", choices=["night", "fog"], default="night")
    parser.add_argument("--no-visual", action="store_true", help="leave frames alone")
    parser.add_argument("--no-text", action="store_true", help="leave report style alone")
    parser.add_argument("--minutes", type=float, default=None, help="switch off after N minutes")
    args = parser.parse_args()
    engine = create_db_engine(get_settings("driftmode").postgres_url)
    if args.action == "status":
        state = read_state(engine)
    else:
        settings = DriftModeSettings(
            preset=args.preset, visual=not args.no_visual, text=not args.no_text
        )
        state = set_state(engine, args.action == "on", settings, duration_minutes=args.minutes)
    print(json.dumps({"active": state.active(), **state.model_dump(mode="json")}, indent=2))


if __name__ == "__main__":
    main()
