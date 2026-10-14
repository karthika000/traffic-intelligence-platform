"""Frame files on the shared volume, and their retention."""

import logging
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

# 20261014T082359.123456Z: sortable, and no ':' so it is also a valid Windows filename.
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"
_TIMESTAMP_RE = re.compile(r"^(\d{8}T\d{6}\.\d{6}Z)\.jpg$")
_TMP_SUFFIX = ".tmp"


def frame_filename(captured_at: datetime) -> str:
    return captured_at.astimezone(UTC).strftime(_TIMESTAMP_FORMAT) + ".jpg"


def parse_frame_timestamp(filename: str) -> datetime | None:
    match = _TIMESTAMP_RE.match(filename)
    if not match:
        return None
    return datetime.strptime(match.group(1), _TIMESTAMP_FORMAT).replace(tzinfo=UTC)


class FrameStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, camera_id: str, captured_at: datetime, data: bytes) -> Path:
        """Write atomically (temp file + rename) so readers never see a partial JPEG."""
        directory = self.root / camera_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / frame_filename(captured_at)
        tmp = path.with_name(path.name + _TMP_SUFFIX)
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return path


def cleanup_expired_frames(root: Path, retention: timedelta, now: datetime | None = None) -> int:
    """Delete frames older than ``retention``. Returns the number of files deleted.

    Age comes from the timestamp in the filename; files with other names (including
    leftover temp files from a crash) fall back to their modification time.
    """
    now = now or datetime.now(UTC)
    cutoff = now - retention
    deleted = 0
    if not root.is_dir():
        return 0
    for camera_dir in root.iterdir():
        if not camera_dir.is_dir():
            continue
        for path in camera_dir.iterdir():
            if not path.is_file():
                continue
            created = parse_frame_timestamp(path.name)
            try:
                if created is None:
                    created = datetime.fromtimestamp(path.stat().st_mtime, UTC)
                if created < cutoff:
                    path.unlink()
                    deleted += 1
            except FileNotFoundError:
                pass  # removed concurrently
            except OSError as exc:
                log.warning("could not delete %s: %s", path, exc)
    return deleted
