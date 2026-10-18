"""Keeps one worker per camera in the database, plus retention cleanup and metrics logging."""

import logging
import threading
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import Engine, select

from traffic_common.db.models import Camera
from traffic_common.driftmode import DriftModeWatcher
from video_ingest.config import IngestConfig
from video_ingest.sources import resolve_source
from video_ingest.storage import FrameStore, cleanup_expired_frames
from video_ingest.worker import CameraWorker, Publish

log = logging.getLogger(__name__)

_JOIN_TIMEOUT = 15.0


@dataclass(frozen=True)
class CameraRow:
    id: str
    stream_url: str


def load_cameras(engine: Engine) -> list[CameraRow]:
    with engine.connect() as conn:
        rows = conn.execute(select(Camera.id, Camera.stream_url).order_by(Camera.id)).all()
    return [CameraRow(r.id, r.stream_url) for r in rows]


class Supervisor:
    def __init__(
        self,
        config: IngestConfig,
        engine: Engine,
        publish: Publish,
        stop: threading.Event,
    ) -> None:
        self.config = config
        self.engine = engine
        self.publish = publish
        self.store = FrameStore(config.frames_dir)
        self._stop = stop
        self.workers: dict[str, CameraWorker] = {}
        self._urls: dict[str, str] = {}
        self.drift_mode = DriftModeWatcher(engine, stop, config.drift_mode_poll_seconds)

    def drift_params(self) -> dict[str, float] | None:
        """Visual degradation while drift mode is on (and not expired), else None."""
        state = self.drift_mode.state
        return state.settings.visual_params() if state.visual_active() else None

    def run(self) -> None:
        """Block until ``stop`` is set."""
        background = [
            threading.Thread(target=self._cleanup_loop, name="frame-cleanup", daemon=True),
            threading.Thread(target=self._metrics_loop, name="metrics", daemon=True),
        ]
        for t in background:
            t.start()
        self.drift_mode.poll()  # before the first frame
        self.drift_mode.start()
        try:
            while not self._stop.is_set():
                self.sync()
                self._stop.wait(self.config.camera_refresh_seconds)
        finally:
            self.stop_all()
            for t in background:
                t.join(timeout=_JOIN_TIMEOUT)

    def sync(self) -> None:
        """Start workers for new cameras, restart changed ones, stop removed ones."""
        try:
            cameras = load_cameras(self.engine)
        except Exception as exc:
            log.error("could not load cameras (keeping %d running): %s", len(self.workers), exc)
            return
        if not cameras and not self.workers:
            log.warning("no cameras in the database; run `python -m traffic_common.db seed`")

        wanted = {c.id: c.stream_url for c in cameras}
        for camera_id in list(self.workers):
            if wanted.get(camera_id) != self._urls[camera_id]:
                reason = "removed" if camera_id not in wanted else "stream_url changed"
                log.info("camera %s: %s, stopping worker", camera_id, reason)
                self._stop_worker(camera_id)
        for camera_id, url in wanted.items():
            if camera_id not in self.workers:
                self._start_worker(camera_id, url)

    def _start_worker(self, camera_id: str, url: str) -> None:
        try:
            source = resolve_source(url, self.config.videos_dir)
        except ValueError as exc:
            log.error("camera %s: invalid stream_url %r: %s", camera_id, url, exc)
            return
        worker = CameraWorker(
            camera_id, source, self.config, self.store, self.publish, drift=self.drift_params
        )
        worker.start()
        self.workers[camera_id] = worker
        self._urls[camera_id] = url
        log.info("camera %s: worker started for %s", camera_id, url)

    def _stop_worker(self, camera_id: str) -> None:
        worker = self.workers.pop(camera_id)
        self._urls.pop(camera_id)
        worker.stop()
        worker.join(timeout=_JOIN_TIMEOUT)

    def stop_all(self) -> None:
        for worker in self.workers.values():
            worker.stop()
        for camera_id in list(self.workers):
            self.workers[camera_id].join(timeout=_JOIN_TIMEOUT)
            if self.workers[camera_id].is_alive():
                log.warning("camera %s: worker did not stop in time", camera_id)
        self.workers.clear()
        self._urls.clear()

    def _cleanup_loop(self) -> None:
        retention = timedelta(minutes=self.config.retention_minutes)
        while not self._stop.wait(self.config.cleanup_interval_seconds):
            try:
                deleted = cleanup_expired_frames(self.config.frames_dir, retention)
                if deleted:
                    log.info("retention: deleted %d frame(s) older than %s", deleted, retention)
            except Exception:
                log.exception("frame cleanup failed")

    def _metrics_loop(self) -> None:
        while not self._stop.wait(self.config.metrics_interval_seconds):
            for worker in list(self.workers.values()):
                log.info("health %s", worker.metrics.report())
