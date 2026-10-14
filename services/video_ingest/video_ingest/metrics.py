"""Per-camera health counters, logged periodically."""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class MetricsSnapshot:
    camera_id: str
    state: str
    frames_read: int
    frames_published: int
    errors: int
    publish_fps: float
    last_error: str | None

    def __str__(self) -> str:
        text = (
            f"camera={self.camera_id} state={self.state} frames_read={self.frames_read} "
            f"frames_published={self.frames_published} fps={self.publish_fps:.2f} "
            f"errors={self.errors}"
        )
        return f"{text} last_error={self.last_error!r}" if self.last_error else text


@dataclass
class CameraMetrics:
    camera_id: str
    state: str = "starting"
    frames_read: int = 0
    frames_published: int = 0
    errors: int = 0
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_report_at: float = field(default_factory=time.monotonic, repr=False)
    _last_report_published: int = field(default=0, repr=False)

    def frame_read(self) -> None:
        with self._lock:
            self.frames_read += 1

    def frame_published(self) -> None:
        with self._lock:
            self.frames_published += 1

    def error(self, exc: BaseException | str) -> None:
        with self._lock:
            self.errors += 1
            self.last_error = str(exc)[:200]

    def set_state(self, state: str) -> None:
        with self._lock:
            self.state = state

    def report(self, now: float | None = None) -> MetricsSnapshot:
        """Snapshot the counters; FPS is frames published per second since the last report."""
        now = time.monotonic() if now is None else now
        with self._lock:
            elapsed = max(now - self._last_report_at, 1e-9)
            fps = (self.frames_published - self._last_report_published) / elapsed
            self._last_report_at = now
            self._last_report_published = self.frames_published
            return MetricsSnapshot(
                self.camera_id, self.state, self.frames_read, self.frames_published,
                self.errors, fps, self.last_error,
            )  # fmt: skip
