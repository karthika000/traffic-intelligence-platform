"""Annotated frames: the latest JPEG per camera, and an MJPEG stream that pushes each new frame
cv_inference writes (it replaces <camera>/latest.jpg atomically, so reads are never partial)."""

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse

router = APIRouter(prefix="/api/v1/frames", tags=["frames"])
NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}
BOUNDARY = "frame"
_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
KEEPALIVE_SECONDS = 10.0  # resend the last frame if nothing new, so proxies keep the stream


def frame_path(request: Request, camera_id: str) -> Path:
    if not _CAMERA_ID.match(camera_id):  # also rules out path traversal
        raise HTTPException(404, f"no frames for {camera_id!r}")
    return request.app.state.config.annotated_dir / camera_id / "latest.jpg"


def _read(path: Path) -> tuple[bytes, float] | None:
    try:
        mtime = path.stat().st_mtime
        return path.read_bytes(), mtime
    except FileNotFoundError:
        return None


@router.get("/{camera_id}/latest.jpg", responses={200: {"content": {"image/jpeg": {}}}})
async def latest_frame(camera_id: str, request: Request) -> Response:
    found = await run_in_threadpool(_read, frame_path(request, camera_id))
    if found is None:
        raise HTTPException(404, f"no frame yet for {camera_id}")
    data, mtime = found
    taken = datetime.fromtimestamp(mtime, UTC).isoformat()
    return Response(data, media_type="image/jpeg", headers={**NO_CACHE, "X-Frame-Time": taken})


@router.get("/{camera_id}/stream.mjpg")
async def mjpeg_stream(
    camera_id: str,
    request: Request,
    fps: float | None = Query(None, gt=0, description="frames per second (server-capped)"),
    limit: int | None = Query(None, ge=1, description="end the stream after this many frames"),
) -> StreamingResponse:
    cfg = request.app.state.config
    path = frame_path(request, camera_id)
    if await run_in_threadpool(_read, path) is None:
        raise HTTPException(404, f"no frame yet for {camera_id}")
    interval = 1.0 / min(fps or cfg.mjpeg_fps, cfg.mjpeg_max_fps)

    async def frames():
        last_mtime, last_sent, sent = None, 0.0, 0
        loop = asyncio.get_running_loop()
        while not await request.is_disconnected():
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                mtime = None
            now = loop.time()
            if mtime is not None and (mtime != last_mtime or now - last_sent > KEEPALIVE_SECONDS):
                found = await run_in_threadpool(_read, path)
                if found is not None:
                    data, last_mtime = found
                    last_sent, sent = now, sent + 1
                    yield (
                        (
                            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(data)}\r\n\r\n"
                        ).encode()
                        + data
                        + b"\r\n"
                    )
                    if limit is not None and sent >= limit:
                        return
            await asyncio.sleep(interval)

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers=NO_CACHE,
    )
