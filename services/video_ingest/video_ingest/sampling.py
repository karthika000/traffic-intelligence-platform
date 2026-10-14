"""Frame sampling and image processing (pure functions, unit tested)."""

import cv2
import numpy as np


class FrameSampler:
    """Decides which frames to keep so that at most ``target_fps`` frames per second pass.

    Works on a monotonically increasing clock in seconds: media time for files
    (frame number / source FPS) or wall-clock time for live streams. Slots are kept on a
    fixed grid, so the output rate doesn't drift; after a gap (a stall or reconnect)
    the grid restarts at the next frame instead of emitting a burst to catch up.
    """

    def __init__(self, target_fps: float) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self.interval = 1.0 / target_fps
        self._next_due: float | None = None

    def should_emit(self, t: float) -> bool:
        # Small tolerance so frame times like 15/30 s hit a 0.5 s slot despite float rounding.
        if self._next_due is not None and t < self._next_due - 1e-6:
            return False
        if self._next_due is None or t - self._next_due >= self.interval:
            self._next_due = t + self.interval  # first frame, or resync after a gap
        else:
            self._next_due += self.interval
        return True


def resize_to_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale so width <= max_width, keeping the aspect ratio. Never upscales."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    new_height = max(1, round(height * max_width / width))
    return cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()
