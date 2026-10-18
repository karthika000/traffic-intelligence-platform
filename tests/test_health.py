"""traffic_common.health: heartbeat files and the healthcheck CLI the compose healthchecks run."""

import json
import time

import pytest

from traffic_common import health


@pytest.fixture(autouse=True)
def health_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALTH_DIR", str(tmp_path))
    return tmp_path


def test_beat_writes_a_heartbeat_the_check_accepts(health_dir) -> None:
    hb = health.Heartbeat("scoring")
    assert health.check("scoring", 60) == "scoring: no heartbeat yet"
    hb.beat()
    data = json.loads((health_dir / "scoring.json").read_text())
    assert data["name"] == "scoring" and abs(data["ts"] - time.time()) < 5
    assert health.check("scoring", 60) is None


def test_stale_heartbeat_fails(health_dir) -> None:
    health.Heartbeat("tick").beat()
    later = time.time() + 120
    assert "last beat" in health.check("tick", 60, now=later)
    assert health.check("tick", 600, now=later) is None


def test_beats_are_throttled_but_state_changes_are_written_at_once(health_dir) -> None:
    hb = health.Heartbeat("consumer-x", min_interval=60)
    hb.beat(assigned=0)
    first = (health_dir / "consumer-x.json").read_text()
    hb.beat(assigned=0)  # nothing new, within the interval: not rewritten
    assert (health_dir / "consumer-x.json").read_text() == first
    hb.beat(assigned=3)
    assert json.loads((health_dir / "consumer-x.json").read_text())["assigned"] == 3


def test_consumer_without_partitions_is_unhealthy(health_dir) -> None:
    hb = health.Heartbeat("consumer-fusion-service")
    hb.beat(assigned=0)
    assert health.check("consumer-fusion-service", 60) == (
        "consumer-fusion-service: no partitions assigned"
    )
    hb.beat(assigned=3)
    assert health.check("consumer-fusion-service", 60) is None


def test_cli_checks_every_heartbeat_with_per_name_max_age(health_dir, capsys) -> None:
    health.Heartbeat("worker").beat()
    health.Heartbeat("consumer-summarizer").beat(assigned=1)
    assert health.main(["worker", "consumer-summarizer"]) == 0
    assert health.main(["worker", "missing"]) == 1
    assert "missing: no heartbeat yet" in capsys.readouterr().out
    old = json.loads((health_dir / "worker.json").read_text())
    old["ts"] -= 300
    (health_dir / "worker.json").write_text(json.dumps(old))
    assert health.main(["worker"]) == 1
    assert health.main(["worker:600"]) == 0


def test_registry_returns_one_heartbeat_per_name() -> None:
    assert health.heartbeat("frames") is health.heartbeat("frames")
