"""Download four openly licensed traffic videos (one per demo camera) into data/videos/
and point each camera's stream_url at its file.

Videos come from Wikimedia Commons. The download URL, licence and author are read
live from the Commons API, so the printed attribution matches the file you get.
The 480p transcodes are used (a few MB each) rather than the 1080p originals.
If a download fails, a synthetic video is generated instead so the pipeline still runs.

Usage (from the repo root, with .venv active and the stack up):
    python scripts/download_sample_videos.py            # download + update the database
    python scripts/download_sample_videos.py --force    # re-download existing files
    python scripts/download_sample_videos.py --synthetic-only
    python scripts/download_sample_videos.py --skip-db  # files only

An attribution record is written to data/videos/ATTRIBUTION.md. CC BY / CC BY-SA
require crediting the author if you redistribute these videos or derived material.
"""

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "video_ingest"))

import cv2  # noqa: E402
from sqlalchemy import update  # noqa: E402

from traffic_common.db import create_db_engine  # noqa: E402
from traffic_common.db.models import Camera  # noqa: E402
from video_ingest.synthetic import generate_traffic_video  # noqa: E402

VIDEOS_DIR = ROOT / "data" / "videos"
CONTAINER_VIDEOS_DIR = "/data/videos"  # where ./data/videos is mounted in video_ingest
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia requires a descriptive User-Agent: https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "traffic-intelligence-platform/0.1 (sample video download script; python-urllib)"
MAX_BYTES = 150 * 1024 * 1024
PREFERRED_HEIGHTS = (720, 480, 360)

CAMERA_VIDEOS = {
    "cam-hyde-park-corner": "File:London roundabout.ogg",
    "cam-trafalgar-square": "File:Intersection Pie-IX-Sherbrooke.webm",
    "cam-elephant-castle": "File:Road traffic on Stritarjeva street.webm",
    "cam-old-street": "File:A intersection with traffic in Chiang Mai, Thailand.ogv",
}


@dataclass
class Result:
    camera_id: str
    path: Path
    source: str
    license: str
    author: str
    page_url: str


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _strip_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()


def commons_info(title: str) -> dict:
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "titles": title, "prop": "videoinfo",
        "viprop": "url|size|extmetadata|derivatives",
    })  # fmt: skip
    pages = _get_json(f"{COMMONS_API}?{query}")["query"]["pages"]
    page = next(iter(pages.values()))
    if "videoinfo" not in page:
        raise RuntimeError(f"{title} not found on Wikimedia Commons")
    return page["videoinfo"][0]


def pick_download(info: dict) -> tuple[str, str]:
    """Prefer a WebM transcode at the first available preferred height; else the original."""
    derivatives = [
        d for d in info.get("derivatives", []) if d.get("type", "").startswith("video/webm")
    ]
    for height in PREFERRED_HEIGHTS:
        for d in derivatives:
            if int(d.get("height", 0)) == height:
                return d["src"], ".webm"
    if info.get("size", 0) > MAX_BYTES:
        raise RuntimeError("no transcode available and the original is too large")
    url = info["url"].split("?")[0]
    return url, Path(urllib.parse.urlparse(url).path).suffix or ".webm"


def download(url: str, dest: Path, attempts: int = 4) -> None:
    """Download with retries on rate limiting (429) and temporary unavailability (503)."""
    for attempt in range(1, attempts + 1):
        try:
            _download_once(url, dest)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == attempts:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else 10 * attempt
            print(f"  HTTP {exc.code}; retrying in {delay}s (attempt {attempt + 1}/{attempts})")
            time.sleep(min(delay, 120))


def _download_once(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_name(dest.name + ".part")
    received = 0
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1024 * 1024):
            received += len(chunk)
            if received > MAX_BYTES:
                raise RuntimeError(f"download exceeds {MAX_BYTES // 2**20} MB")
            out.write(chunk)
    tmp.replace(dest)


def readable(path: Path) -> bool:
    """True if OpenCV can open the file and decode at least one frame."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        ok, _ = cap.read()
        return bool(cap.isOpened() and ok)
    finally:
        cap.release()


def fetch_camera_video(camera_id: str, title: str, force: bool) -> Result:
    info = commons_info(title)
    meta = info.get("extmetadata", {})
    license_name = _strip_html(meta.get("LicenseShortName", {}).get("value", "unknown"))
    author = _strip_html(meta.get("Artist", {}).get("value", "unknown"))
    page_url = info.get("descriptionurl", "")
    url, suffix = pick_download(info)
    dest = VIDEOS_DIR / f"{camera_id}{suffix}"
    if force or not dest.exists() or not readable(dest):
        print(f"  downloading {url}")
        download(url, dest)
    else:
        print(f"  already present: {dest.name}")
    if not readable(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError("downloaded file could not be decoded by OpenCV")
    return Result(camera_id, dest, title, license_name, author, page_url)


def synthetic_video(camera_id: str, seed: int) -> Result:
    dest = VIDEOS_DIR / f"{camera_id}-synthetic.mp4"
    generate_traffic_video(dest, seconds=20, seed=seed)
    return Result(
        camera_id,
        dest,
        "synthetic (generated with OpenCV)",
        "n/a (generated locally)",
        "this script",
        "",
    )


def update_database(results: list[Result], db_url: str | None) -> None:
    engine = create_db_engine(db_url)
    try:
        with engine.begin() as conn:
            for r in results:
                url = f"{CONTAINER_VIDEOS_DIR}/{r.path.name}"
                count = conn.execute(
                    update(Camera).where(Camera.id == r.camera_id).values(stream_url=url)
                ).rowcount
                status = "updated" if count else "NOT FOUND (run: python -m traffic_common.db seed)"
                print(f"  {r.camera_id:<22} stream_url={url}  {status}")
    finally:
        engine.dispose()


def write_attribution(results: list[Result]) -> Path:
    lines = ["# Sample video attribution", ""]
    for r in results:
        lines += [
            f"## {r.path.name} ({r.camera_id})",
            f"- Source: {r.source}",
            f"- Author: {r.author}",
            f"- Licence: {r.license}",
        ]
        if r.page_url:
            lines.append(f"- File page: {r.page_url}")
        lines.append("")
    path = VIDEOS_DIR / "ATTRIBUTION.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--synthetic-only", action="store_true", help="skip downloads")
    parser.add_argument("--skip-db", action="store_true", help="don't update cameras.stream_url")
    parser.add_argument("--db-url", help="database URL (default: POSTGRES_URL)")
    args = parser.parse_args()

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    for seed, (camera_id, title) in enumerate(CAMERA_VIDEOS.items()):
        print(f"{camera_id}: {title}")
        try:
            if args.synthetic_only:
                raise RuntimeError("--synthetic-only")
            results.append(fetch_camera_video(camera_id, title, args.force))
        except Exception as exc:
            print(f"  download failed ({exc}); generating a synthetic video instead")
            results.append(synthetic_video(camera_id, seed))

    print("\nVideos:")
    for r in results:
        size_mb = r.path.stat().st_size / 2**20
        print(f"  {r.path.name:<34} {size_mb:6.1f} MB  {r.license:<24} {r.author}")
        print(f"  {'':<34} source: {r.source}" + (f"  ({r.page_url})" if r.page_url else ""))
    print(f"\nAttribution written to {write_attribution(results).relative_to(ROOT)}")

    if not args.skip_db:
        print("\nUpdating cameras.stream_url:")
        update_database(results, args.db_url)
        print("video_ingest picks these up within CAMERA_REFRESH_SECONDS (default 60).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
