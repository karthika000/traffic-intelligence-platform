"""Executes one scenario run against one camera, emitting tagged reports and events."""

import logging
import math
import random
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo

from event_simulator.emitter import Emitter
from event_simulator.groundtruth import GroundTruthStore
from event_simulator.locations import CameraInfo
from event_simulator.patterns import RoadProfile, WeatherModel, sensor_reading
from event_simulator.reports import ReportGenerator, Variant
from event_simulator.scenarios import PhaseWindow, Scenario
from traffic_common.enums import ScenarioStatus

log = logging.getLogger(__name__)


@dataclass
class CameraState:
    """Per-camera simulation state. While a scenario runs, its runner owns the camera."""

    info: CameraInfo
    weather: WeatherModel
    road: RoadProfile = field(default_factory=RoadProfile)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def weather_payload(state: CameraState, local_now: datetime) -> dict:
    with state.lock:
        state.weather.tick(local_now)
        return state.weather.as_payload()


def sensor_payload(
    state: CameraState, local_now: datetime, interval: float, rng: random.Random,
    speed_factor: float = 1.0, count_factor: float = 1.0,
) -> dict:  # fmt: skip
    r = sensor_reading(local_now, interval, rng, state.road, count_factor, speed_factor)
    return {
        "detector_id": f"{state.info.id}-loop-1",
        "vehicle_count": r.vehicle_count,
        "flow_veh_per_hour": r.flow_veh_per_hour,
        "avg_speed_kmh": r.avg_speed_kmh,
        "occupancy": r.occupancy,
        "interval_seconds": r.interval_seconds,
    }


class ScenarioRunner(threading.Thread):
    def __init__(
        self,
        run_id: uuid.UUID,
        scenario: Scenario,
        camera: CameraState,
        emitter: Emitter,
        store: GroundTruthStore | None,
        *,
        time_scale: float = 1.0,
        sensor_interval_seconds: float = 15.0,
        local_tz: tzinfo = UTC,
        triggered_by: str = "cli",
        rng: random.Random | None = None,
        on_finish: Callable[["ScenarioRunner"], None] | None = None,
        text_style: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__(name=f"scenario-{scenario.name}-{camera.info.id}", daemon=True)
        self.run_id = run_id
        self.scenario = scenario
        self.camera = camera
        self.emitter = emitter
        self.store = store
        self.time_scale = time_scale
        self.sensor_interval = sensor_interval_seconds / time_scale
        self.local_tz = local_tz
        self.triggered_by = triggered_by
        self.rng = rng or random.Random()
        self.generator = ReportGenerator(self.rng, Variant.LIVE, text_style)
        self.on_finish = on_finish
        self.timeline = scenario.timeline(time_scale)
        self.status = ScenarioStatus.RUNNING
        self.reports_emitted = 0
        self.events_emitted = 0
        self.current_phase: str | None = None
        self._stop_event = threading.Event()  # not "_stop": a threading.Thread internal

    @property
    def duration_seconds(self) -> float:
        return self.timeline[-1].end

    def cancel(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        started_at = datetime.now(UTC)
        error: str | None = None
        log.info(
            "scenario %s started: %s on %s (time_scale %.1f, %.0fs, %d phases)",
            self.run_id, self.scenario.name, self.camera.info.id, self.time_scale,
            self.duration_seconds, len(self.timeline),
        )  # fmt: skip
        try:
            self._record_start(started_at)
            self._execute()
            self.status = (
                ScenarioStatus.CANCELLED if self._stop_event.is_set() else ScenarioStatus.COMPLETED
            )
        except Exception as exc:
            self.status, error = ScenarioStatus.FAILED, str(exc)
            log.exception("scenario %s failed", self.run_id)
        finally:
            with self.camera.lock:
                self.camera.weather.override = None
            self._record_finish(error)
            log.info(
                "scenario %s %s: %d reports, %d events", self.run_id, self.status.value,
                self.reports_emitted, self.events_emitted,
            )  # fmt: skip
            if self.on_finish:
                self.on_finish(self)

    def _execute(self) -> None:
        t0 = time.monotonic()
        window: PhaseWindow | None = None
        next_sensor = 0.0
        next_report = math.inf
        while not self._stop_event.is_set():
            t = time.monotonic() - t0
            if t >= self.duration_seconds:
                return
            current = next(w for w in self.timeline if w.contains(t) or w is self.timeline[-1])
            if current is not window:
                window = current
                next_report = self._enter_phase(window, t)

            if t >= next_sensor:
                self._emit_readings(window, t)
                next_sensor += self.sensor_interval
            if t >= next_report:
                self._emit_report(window)
                next_report = t + self._report_gap(window)

            wake_at = min(next_sensor, next_report, window.end)
            self._stop_event.wait(max(0.0, wake_at - (time.monotonic() - t0)))

    def _enter_phase(self, window: PhaseWindow, t: float) -> float:
        phase = window.phase
        self.current_phase = phase.name
        log.info(
            "scenario %s phase %s (severity %s) at +%.1fs",
            self.run_id, phase.name, phase.severity.value, t,
        )  # fmt: skip
        with self.camera.lock:
            self.camera.weather.override = phase.weather.as_override() if phase.weather else None
        return t + self._report_gap(window)

    def _report_gap(self, window: PhaseWindow) -> float:
        reports = window.phase.reports
        if reports is None or reports.per_minute <= 0:
            return math.inf
        rate_per_second = reports.per_minute * self.time_scale / 60
        return self.rng.expovariate(rate_per_second)  # Poisson arrivals

    def _emit_report(self, window: PhaseWindow) -> None:
        phase = window.phase
        spec = phase.reports
        assert spec is not None
        types = self.scenario.report_incident_types(phase)
        incident_type = self.rng.choices(list(types), weights=list(types.values()))[0]
        style = self.rng.choices(list(spec.sources), weights=list(spec.sources.values()))[0]
        report = self.generator.generate(
            self.camera.info, incident_type=incident_type, severity=phase.severity,
            style=style, stage=spec.stage, now=datetime.now(self.local_tz),
        )  # fmt: skip
        self.emitter.report(self.camera.info, report, scenario_id=self.run_id, phase=phase.name)
        self.reports_emitted += 1

    def _emit_readings(self, window: PhaseWindow, t: float) -> None:
        local_now = datetime.now(self.local_tz)
        speed_factor, count_factor = window.traffic_factors(t)
        info = self.camera.info
        self.emitter.event(info, "weather", weather_payload(self.camera, local_now), self.run_id)
        sensor = sensor_payload(
            self.camera, local_now, self.sensor_interval * self.time_scale, self.rng,
            speed_factor, count_factor,
        )  # fmt: skip
        self.emitter.event(info, "loop_detector", sensor, self.run_id)
        self.events_emitted += 2

    def _record_start(self, started_at: datetime) -> None:
        if self.store is None:
            return
        timeline = []
        for w in self.timeline:
            record = w.as_record()
            record["starts_at"] = (started_at + timedelta(seconds=w.start)).isoformat()
            record["ends_at"] = (started_at + timedelta(seconds=w.end)).isoformat()
            timeline.append(record)
        self.store.start_run(
            self.run_id, self.scenario, self.camera.info.id, self.triggered_by, self.time_scale,
            started_at, started_at + timedelta(seconds=self.duration_seconds), timeline,
        )  # fmt: skip

    def _record_finish(self, error: str | None) -> None:
        if self.store is None:
            return
        try:
            self.store.finish_run(
                self.run_id, self.status, datetime.now(UTC), self.reports_emitted,
                self.events_emitted, error,
            )  # fmt: skip
        except Exception as exc:
            log.error("could not record end of scenario %s: %s", self.run_id, exc)
