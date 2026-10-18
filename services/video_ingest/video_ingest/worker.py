"""One thread per camera: read, sample, resize, encode, store and publish frames."""

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from traffic_common.schemas import FrameMessage
from video_ingest.config import IngestConfig
from video_ingest.degrade import degrade
from video_ingest.metrics import CameraMetrics
from video_ingest.sampling import FrameSampler, encode_jpeg, resize_to_max_width
from video_ingest.sources import SourceError, VideoSource, open_capture, source_fps
from video_ingest.storage import FrameStore

log = logging.getLogger(__name__)

Publish = Callable[[FrameMessage], None]
# Drift mode's visual parameters while it is on, else None (traffic_common.driftmode).
DriftParams = Callable[[], dict[str, float] | None]


class CameraWorker(threading.Thread):
    def __init__(
        self,
        camera_id: str,
        source: VideoSource,
        config: IngestConfig,
        store: FrameStore,
        publish: Publish,
        metrics: CameraMetrics | None = None,
        drift: DriftParams | None = None,
    ) -> None:
        super().__init__(name=f"camera-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.config = config
        self.store = store
        self.publish = publish
        self.metrics = metrics or CameraMetrics(camera_id)
        self.drift = drift
        self._rng = np.random.default_rng()
        self.sequence = 0
        # Not "_stop": that name is an internal threading.Thread method.
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        delay = self.config.retry_initial_seconds
        while not self._stop_event.is_set():
            published_before = self.metrics.frames_published
            try:
                self.metrics.set_state("connecting")
                self._stream()
                if self._stop_event.is_set():
                    break
                reason = "stream ended"
            except Exception as exc:  # keep the worker alive whatever the source does
                self.metrics.error(exc)
                reason = str(exc)
            if self.metrics.frames_published > published_before:
                delay = self.config.retry_initial_seconds  # the session was healthy
            self.metrics.set_state("retrying")
            log.warning("camera %s: %s; retrying in %.1fs", self.camera_id, reason, delay)
            if self._stop_event.wait(delay):
                break
            delay = min(delay * 2, self.config.retry_max_seconds)
        self.metrics.set_state("stopped")

    def _stream(self) -> None:
        """Read until the source ends (live) or forever (looping file). Raises SourceError."""
        cap = open_capture(self.source, self.config.source_timeout_seconds)
        try:
            fps = source_fps(cap)
            sampler = FrameSampler(self.config.sample_fps)
            log.info(
                "camera %s: streaming %s %s at %.2f fps (sampling %.2f fps)",
                self.camera_id, self.source.kind, self.source.location, fps,
                self.config.sample_fps,
            )  # fmt: skip
            self.metrics.set_state("streaming")
            wall_start = time.monotonic()
            media_offset = 0.0  # seconds of video played in earlier loops of a file
            frames_this_pass = 0

            while not self._stop_event.is_set():
                if not cap.grab():
                    if not self.source.loops:
                        return
                    if frames_this_pass == 0:
                        raise SourceError(f"no frames could be read from {self.source.location}")
                    # Loop the file: reopen rather than seek, which is unreliable for some codecs.
                    media_offset += frames_this_pass / fps
                    frames_this_pass = 0
                    cap.release()
                    cap = open_capture(self.source, self.config.source_timeout_seconds)
                    continue

                frames_this_pass += 1
                self.metrics.frame_read()
                if self.source.loops:
                    # Files would otherwise be read as fast as they decode: play them in real time.
                    t = media_offset + frames_this_pass / fps
                    ahead = wall_start + t - time.monotonic()
                    if ahead > 0 and self._stop_event.wait(ahead):
                        return
                else:
                    t = time.monotonic()

                if sampler.should_emit(t):
                    ok, frame = cap.retrieve()  # decode only the frames we keep
                    if not ok or frame is None:
                        self.metrics.error("could not decode frame")
                        continue
                    self._emit(frame, fps)
        finally:
            cap.release()

    def _emit(self, frame: np.ndarray, fps: float) -> None:
        captured_at = datetime.now(UTC)
        try:
            frame = resize_to_max_width(frame, self.config.max_width)
            drift = self.drift() if self.drift else None
            if drift:
                frame = degrade(frame, drift, self._rng)
            path = self.store.write(
                self.camera_id, captured_at, encode_jpeg(frame, self.config.jpeg_quality)
            )
            height, width = frame.shape[:2]
            self.publish(
                FrameMessage(
                    camera_id=self.camera_id,
                    timestamp=captured_at,
                    sequence=self.sequence,
                    frame_path=path.as_posix(),
                    width=width,
                    height=height,
                    source_fps=fps,
                )
            )
        except Exception as exc:  # one bad frame shouldn't stop the stream
            self.metrics.error(exc)
            log.warning("camera %s: failed to emit frame: %s", self.camera_id, exc)
            return
        self.sequence += 1
        self.metrics.frame_published()
