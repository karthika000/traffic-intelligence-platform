"""Resolve a camera's stream_url and open it with OpenCV."""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import cv2

# Prefer TCP for RTSP: UDP drops packets (and frames) on busy networks and through Docker NAT.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# Some containers/streams report 0, NaN or nonsense frame rates.
FALLBACK_FPS = 25.0
_MAX_PLAUSIBLE_FPS = 240.0


class SourceKind(StrEnum):
    FILE = "file"
    RTSP = "rtsp"
    HTTP = "http"


@dataclass(frozen=True)
class VideoSource:
    kind: SourceKind
    location: str  # filesystem path or URL, as passed to OpenCV

    @property
    def loops(self) -> bool:
        """Local files replay forever to simulate a live feed."""
        return self.kind is SourceKind.FILE


class SourceError(RuntimeError):
    """The source could not be opened or stopped producing frames."""


def resolve_source(stream_url: str, videos_dir: Path) -> VideoSource:
    """Map a cameras.stream_url to a source.

    Accepted forms: ``rtsp://`` / ``rtsps://``, ``http://`` / ``https://``,
    ``file:///data/videos/x.mp4``, an absolute path, or a path relative to ``videos_dir``.
    """
    url = stream_url.strip()
    if not url:
        raise ValueError("empty stream_url")
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme in ("rtsp", "rtsps"):
        return VideoSource(SourceKind.RTSP, url)
    if scheme in ("http", "https"):
        return VideoSource(SourceKind.HTTP, url)
    if scheme == "file":
        return VideoSource(SourceKind.FILE, unquote(parsed.path))
    if scheme and len(scheme) > 1:  # a single letter is a Windows drive, e.g. C:\videos\x.mp4
        raise ValueError(f"unsupported stream_url scheme {scheme!r}: {stream_url}")
    if PurePosixPath(url).is_absolute() or Path(url).is_absolute():
        return VideoSource(SourceKind.FILE, url)
    return VideoSource(SourceKind.FILE, str(videos_dir / url))


def open_capture(source: VideoSource, timeout_seconds: float) -> cv2.VideoCapture:
    if source.kind is SourceKind.FILE and not Path(source.location).is_file():
        raise SourceError(f"video file not found: {source.location}")
    timeout_ms = int(timeout_seconds * 1000)
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
    ]  # fmt: skip
    cap = cv2.VideoCapture(source.location, cv2.CAP_FFMPEG, params)
    if not cap.isOpened():
        cap.release()
        raise SourceError(f"could not open {source.kind} source {source.location}")
    return cap


def source_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0 or fps > _MAX_PLAUSIBLE_FPS:  # fps != fps: NaN
        return FALLBACK_FPS
    return float(fps)
