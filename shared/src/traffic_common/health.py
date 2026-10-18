"""Heartbeats for container healthchecks: "healthy" means the service is doing its work, not
just that the process exists.

Each piece of work a service does (a consumer's poll loop, a scoring loop, publishing frames)
owns a named heartbeat and calls ``beat()`` every time it makes progress. A beat writes
``<HEALTH_DIR>/<name>.json`` (at most every ``min_interval`` seconds, or at once when the state
changes), with the time and any state passed in. Consumers record how many partitions they
are assigned, so a consumer that is polling but owns no partitions is not healthy.

The compose healthcheck runs the check CLI, which fails when a heartbeat is missing, older
than ``--max-age`` seconds, or reports ``assigned: 0``:

    python -m traffic_common.health consumer-fusion-service tick --max-age 60
    python -m traffic_common.health frames:30 --max-age 60     # per-heartbeat max age
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_MIN_INTERVAL = 2.0


def health_dir() -> Path:
    return Path(os.getenv("HEALTH_DIR") or Path(tempfile.gettempdir()) / "health")


class Heartbeat:
    def __init__(self, name: str, min_interval: float = DEFAULT_MIN_INTERVAL) -> None:
        self.name = name
        self.min_interval = min_interval
        self._state: dict[str, Any] = {}
        self._last_write = 0.0
        self._lock = threading.Lock()
        self.failed = False  # writing failed once (read-only /tmp?); logged by the caller

    @property
    def path(self) -> Path:
        return health_dir() / f"{self.name}.json"

    def beat(self, **state: Any) -> None:
        """Record progress. Cheap enough to call on every loop iteration."""
        now = time.time()
        with self._lock:
            changed = bool(state) and any(self._state.get(k) != v for k, v in state.items())
            self._state.update(state)
            if not changed and now - self._last_write < self.min_interval:
                return
            self._last_write = now
            body = json.dumps({"name": self.name, "ts": now, **self._state})
        try:
            path = self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, path)  # atomic: the check never reads a half-written file
        except OSError:
            self.failed = True


_registry: dict[str, Heartbeat] = {}
_registry_lock = threading.Lock()


def heartbeat(name: str) -> Heartbeat:
    """The process-wide heartbeat called ``name``."""
    with _registry_lock:
        hb = _registry.get(name)
        if hb is None:
            hb = _registry[name] = Heartbeat(name)
        return hb


def check(name: str, max_age: float, now: float | None = None) -> str | None:
    """None if the heartbeat is fresh and working, otherwise why not."""
    path = health_dir() / f"{name}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"{name}: no heartbeat yet"
    except (OSError, ValueError) as exc:
        return f"{name}: unreadable heartbeat ({exc})"
    age = (now or time.time()) - float(data.get("ts", 0))
    if age > max_age:
        return f"{name}: last beat {age:.0f}s ago (max {max_age:.0f}s)"
    if data.get("assigned") == 0:
        return f"{name}: no partitions assigned"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m traffic_common.health")
    parser.add_argument("names", nargs="+", help="heartbeat names, optionally name:max_age")
    parser.add_argument("--max-age", type=float, default=60.0, help="seconds (default 60)")
    args = parser.parse_args(argv)
    problems = []
    for spec in args.names:
        name, _, age = spec.partition(":")
        problem = check(name, float(age) if age else args.max_age)
        if problem:
            problems.append(problem)
    if problems:
        print("; ".join(problems))
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
