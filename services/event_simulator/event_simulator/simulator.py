"""Background generation for every camera, plus scenario management."""

import logging
import random
import threading
import uuid
import zlib
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import Engine, select

from event_simulator.config import SimulatorConfig
from event_simulator.emitter import Emitter
from event_simulator.groundtruth import GroundTruthStore
from event_simulator.locations import CameraInfo
from event_simulator.patterns import RoadProfile, WeatherModel
from event_simulator.reports import ReportGenerator, Variant
from event_simulator.runner import CameraState, ScenarioRunner, sensor_payload, weather_payload
from event_simulator.scenarios import Scenario, load_scenarios
from traffic_common.db.models import Camera
from traffic_common.driftmode import DriftModeWatcher
from traffic_common.schemas import SimulatorCommand

log = logging.getLogger(__name__)
_JOIN_TIMEOUT = 15.0


def load_cameras(engine: Engine) -> list[CameraInfo]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(Camera.id, Camera.name, Camera.location_name, Camera.latitude, Camera.longitude)
            .order_by(Camera.id)
        ).all()  # fmt: skip
    return [CameraInfo(r.id, r.name, r.location_name, r.latitude, r.longitude) for r in rows]


def _road_profile(camera_id: str) -> RoadProfile:
    """Stable per-camera variation so cameras don't all report identical traffic."""
    h = zlib.crc32(camera_id.encode())
    return RoadProfile(
        capacity_veh_per_hour=1400 + h % 1200,
        free_flow_kmh=40 + (h >> 8) % 16,
    )


class Simulator:
    def __init__(
        self,
        config: SimulatorConfig,
        engine: Engine | None,
        emitter: Emitter,
        store: GroundTruthStore | None,
        stop: threading.Event,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.emitter = emitter
        self.store = store
        self.rng = rng or random.Random()
        # Drift mode (demo) switches every report, background and scenario, to a new style.
        self.drift_mode = DriftModeWatcher(engine, stop, config.drift_mode_poll_seconds)
        self.generator = ReportGenerator(self.rng, Variant.LIVE, self.text_style)
        self.tz = ZoneInfo(config.timezone)
        self.scenarios: dict[str, Scenario] = load_scenarios(config.scenarios_dir)
        self.cameras: dict[str, CameraState] = {}
        self.runs: dict[str, ScenarioRunner] = {}  # camera_id -> active run
        self._lock = threading.Lock()
        self._stop = stop
        self._weather_seed = self.rng.random()

    def text_style(self) -> str | None:
        return self.drift_mode.state.text_style()

    # ------------------------------------------------------------------ lifecycle

    def run(self) -> None:
        """Block until ``stop`` is set."""
        loops = [self._report_loop, self._sensor_loop, self._stats_loop]
        if self.config.scenario_schedule_minutes > 0:
            loops.append(self._schedule_loop)
        threads = [
            threading.Thread(target=f, name=f.__name__.strip("_"), daemon=True) for f in loops
        ]
        log.info("scenarios available: %s", ", ".join(sorted(self.scenarios)) or "none")
        try:
            self.refresh_cameras()
            if self.engine is not None:
                self.drift_mode.poll()
                self.drift_mode.start()
            for t in threads:
                t.start()
            while not self._stop.wait(self.config.camera_refresh_seconds):
                self.refresh_cameras()
        finally:
            for runner in list(self.runs.values()):
                runner.cancel()
            for runner in list(self.runs.values()):
                runner.join(timeout=_JOIN_TIMEOUT)
            for t in threads:
                t.join(timeout=_JOIN_TIMEOUT)

    def refresh_cameras(self) -> None:
        if self.engine is None:
            return
        try:
            cameras = load_cameras(self.engine)
        except Exception as exc:
            log.error("could not load cameras (keeping %d): %s", len(self.cameras), exc)
            return
        with self._lock:
            wanted = {c.id: c for c in cameras}
            for camera_id in set(self.cameras) - set(wanted):
                log.info("camera %s removed", camera_id)
                del self.cameras[camera_id]
            for camera_id, info in wanted.items():
                state = self.cameras.get(camera_id)
                if state is None:
                    self.cameras[camera_id] = self._new_state(info)
                    log.info("camera %s added (%s)", camera_id, info.location_name)
                elif state.info != info:
                    state.info = info
        if not cameras:
            log.warning("no cameras in the database; run `python -m traffic_common.db seed`")

    def add_camera(self, info: CameraInfo) -> None:
        """For tests and tools that run without a database."""
        with self._lock:
            self.cameras[info.id] = self._new_state(info)

    def _new_state(self, info: CameraInfo) -> CameraState:
        # Each camera's weather evolves independently from its own seed.
        weather = WeatherModel(random.Random(f"{self._weather_seed}-{info.id}"))
        return CameraState(info=info, weather=weather, road=_road_profile(info.id))

    # ------------------------------------------------------------------ scenarios

    def start_scenario(
        self,
        name: str,
        camera_id: str,
        *,
        time_scale: float = 1.0,
        run_id: uuid.UUID | None = None,
        triggered_by: str = "cli",
    ) -> ScenarioRunner:
        scenario = self.scenarios.get(name)
        if scenario is None:
            raise ValueError(f"unknown scenario {name!r}; available: {sorted(self.scenarios)}")
        with self._lock:
            camera = self.cameras.get(camera_id)
            if camera is None:
                raise ValueError(f"unknown camera {camera_id!r}")
            if camera_id in self.runs:
                raise ValueError(
                    f"camera {camera_id} already runs scenario {self.runs[camera_id].run_id}"
                )
            runner = ScenarioRunner(
                run_id or uuid.uuid4(), scenario, camera, self.emitter, self.store,
                time_scale=time_scale, sensor_interval_seconds=self.config.sensor_interval_seconds,
                local_tz=self.tz, triggered_by=triggered_by,
                rng=random.Random(self.rng.random()), on_finish=self._run_finished,
                text_style=self.text_style,
            )  # fmt: skip
            self.runs[camera_id] = runner
        runner.start()
        return runner

    def stop_scenario(self, run_id: uuid.UUID) -> bool:
        with self._lock:
            runner = next((r for r in self.runs.values() if r.run_id == run_id), None)
        if runner is None:
            return False
        runner.cancel()
        return True

    def _run_finished(self, runner: ScenarioRunner) -> None:
        with self._lock:
            if self.runs.get(runner.camera.info.id) is runner:
                del self.runs[runner.camera.info.id]

    def handle_command(self, command: SimulatorCommand) -> None:
        """Handler for simulator.control. Invalid commands are logged and dropped."""
        age = (datetime.now(UTC) - command.timestamp).total_seconds()
        if age > self.config.max_command_age_seconds:
            log.warning("ignoring %s from %.0fs ago (stale)", command.action, age)
            return
        if command.action == "stop_scenario":
            found = self.stop_scenario(command.scenario_id)
            log.info("stop %s: %s", command.scenario_id, "cancelling" if found else "not running")
            return
        try:
            self.start_scenario(
                command.scenario or "", command.camera_id or "", time_scale=command.time_scale,
                run_id=command.scenario_id, triggered_by=command.requested_by,
            )  # fmt: skip
        except ValueError as exc:
            log.error("rejected start_scenario %s: %s", command.scenario_id, exc)

    # ------------------------------------------------------------------ loops

    def _report_loop(self) -> None:
        while not self._stop.wait(self.rng.expovariate(1 / self.config.report_interval_seconds)):
            with self._lock:
                states = list(self.cameras.values())
            if not states:
                continue
            state = self.rng.choice(states)
            try:
                report = self.generator.generate(state.info, now=datetime.now(self.tz))
                self.emitter.report(state.info, report)
            except Exception:
                log.exception("background report failed")

    def _sensor_loop(self) -> None:
        interval = self.config.sensor_interval_seconds
        while not self._stop.wait(interval):
            with self._lock:
                idle = [s for cid, s in self.cameras.items() if cid not in self.runs]
            local_now = datetime.now(self.tz)
            for state in idle:  # cameras running a scenario get readings from their runner
                try:
                    self.emitter.event(state.info, "weather", weather_payload(state, local_now))
                    payload = sensor_payload(state, local_now, interval, self.rng)
                    self.emitter.event(state.info, "loop_detector", payload)
                except Exception:
                    log.exception("sensor tick failed for %s", state.info.id)

    def _schedule_loop(self) -> None:
        period = self.config.scenario_schedule_minutes * 60
        log.info("random scenarios scheduled every %.1f min", self.config.scenario_schedule_minutes)
        while not self._stop.wait(period):
            with self._lock:
                free = [cid for cid in self.cameras if cid not in self.runs]
            if not free or not self.scenarios:
                continue
            try:
                self.start_scenario(
                    self.rng.choice(sorted(self.scenarios)), self.rng.choice(free),
                    time_scale=self.config.scheduled_time_scale, triggered_by="schedule",
                )  # fmt: skip
            except Exception:
                log.exception("scheduled scenario failed to start")

    def _stats_loop(self) -> None:
        while not self._stop.wait(self.config.stats_interval_seconds):
            s = self.emitter.stats
            with self._lock:
                active = [
                    f"{r.scenario.name}@{cid}[{r.current_phase}]" for cid, r in self.runs.items()
                ]
                cameras = len(self.cameras)
            log.info(
                "stats cameras=%d reports=%d events=%d db_errors=%d send_errors=%d active=%s",
                cameras, s.reports, s.events, s.db_errors, s.send_errors, active or "none",
            )  # fmt: skip
