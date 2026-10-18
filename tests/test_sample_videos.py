"""scripts/download_sample_videos.py: transient network failures are retried, permanent ones
are not (the camera then falls back to a synthetic video)."""

import importlib.util
import ssl
import sys
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def script(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "download_sample_videos", ROOT / "scripts" / "download_sample_videos.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "download_sample_videos", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    return module


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", Message(), None)


def flaky(*failures):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return "ok"

    return fn, calls


def test_transient_failures_are_retried(script) -> None:
    fn, calls = flaky(
        http_error(429),
        urllib.error.URLError(ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")),
        TimeoutError("read timed out"),
        http_error(503),
    )
    assert script.with_retries(fn, "download") == "ok"
    assert len(calls) == 5


def test_permanent_errors_and_exhausted_retries_raise(script) -> None:
    fn, calls = flaky(http_error(404))
    with pytest.raises(urllib.error.HTTPError):
        script.with_retries(fn, "download")
    assert len(calls) == 1
    fn, calls = flaky(*[http_error(429)] * 5)
    with pytest.raises(urllib.error.HTTPError):
        script.with_retries(fn, "download", attempts=5)
    assert len(calls) == 5
