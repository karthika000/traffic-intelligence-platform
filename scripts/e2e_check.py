"""End-to-end check of the running platform, through the API only (standard library only, so
it runs anywhere with Python 3.11; scripts/e2e_test.ps1 runs it in the api image).

Waits for anomaly_service's camera warm-up (--prometheus) and for cameras without an open
incident, triggers every scenario at once, each on a different camera, then follows each run
until its incident has resolved or the timeout is reached. A run passes when, for the
incident carrying the run's reports (matched by the reports' scenario_id):

- confirmed:   it reached "confirmed" (confirmed_at is set)
- type:        its incident type is the scenario's
- summary:     at least one summary was written for it
- api:         GET /api/v1/incidents lists it and GET /api/v1/incidents/{id} returns it
- resolved:    it reached "resolved"

It prints a table with the timings (from the trigger) and how far the incident's start_time
is from the run's actual start, writes the results as JSON (--json), and exits 1 on failure.

    python scripts/e2e_check.py --api http://localhost:8001 --time-scale 6
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

CHECKS = ("confirmed", "type", "summary", "api", "resolved")


def ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def request(self, method: str, path: str, body: dict | None = None, **query):
        url = self.base + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)

    def get(self, path: str, **query):
        return self.request("GET", path, **query)

    def post(self, path: str, body: dict):
        return self.request("POST", path, body)


@dataclass
class Run:
    scenario: str
    camera_id: str
    expected_type: str
    duration_s: float
    scenario_id: str = ""
    triggered_at: datetime | None = None
    run_started_at: datetime | None = None
    run_ended_at: datetime | None = None
    run_status: str = ""
    incident_id: str | None = None
    incident_type: str | None = None
    status: str | None = None
    start_time: datetime | None = None
    first_evidence_at: datetime | None = None
    suspected_at: datetime | None = None
    confirmed_at: datetime | None = None
    summary_at: datetime | None = None
    end_time: datetime | None = None
    summaries: int = 0
    listed: bool = False
    fetched: bool = False
    candidates: list[str] = field(default_factory=list)  # incidents with this run's reports

    def results(self) -> dict[str, bool]:
        return {
            "confirmed": self.confirmed_at is not None,
            "type": self.incident_type == self.expected_type,
            "summary": self.summaries > 0,
            "api": self.listed and self.fetched,
            "resolved": self.status == "resolved",
        }

    @property
    def passed(self) -> bool:
        return all(self.results().values())

    @property
    def done(self) -> bool:
        """Nothing more to wait for: passed, or resolved a minute ago (its last summary had
        time to arrive) without passing."""
        if self.passed:
            return True
        return self.end_time is not None and datetime.now(UTC) - self.end_time > timedelta(
            seconds=60
        )

    def after(self, moment: datetime | None) -> str:
        if moment is None or self.triggered_at is None:
            return "-"
        s = (moment - self.triggered_at).total_seconds()
        return f"{int(s // 60)}m{int(s % 60):02d}s"


def pick_cameras(cameras: list[dict], n: int) -> list[str]:
    """n different cameras, preferring ones without an open incident (fusion would attach the
    run's reports to it instead of opening a new one)."""
    free = sorted(c["id"] for c in cameras if not c.get("open_incidents"))
    ids = free if len(free) >= n else sorted(c["id"] for c in cameras)
    if len(ids) < n:
        raise SystemExit(f"need {n} cameras for {n} scenarios, found {len(ids)}")
    return ids[:n]


def wait_for_warm_up(prometheus: str, minutes: float) -> None:
    """Wait up to ``minutes`` until anomaly_service reports no camera in its warm-up (it raises
    no traffic alerts before then), going ahead if Prometheus can't be reached."""
    if not prometheus:
        return
    url = prometheus.rstrip("/") + "/api/v1/query?query=traffic_cameras_warming_up"
    deadline = time.monotonic() + minutes * 60
    while True:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                result = json.load(resp)["data"]["result"]
            warming = int(float(result[0]["value"][1])) if result else None
        except (OSError, ValueError, KeyError, IndexError) as exc:
            print(f"warm-up state unavailable ({exc}); not waiting for it", flush=True)
            return
        if warming == 0 or time.monotonic() >= deadline:
            return
        print(f"waiting for anomaly_service's warm-up: {warming} camera(s) left", flush=True)
        time.sleep(15)


def wait_for_cameras(api: Api, n: int, minutes: float) -> list[str]:
    """Wait up to ``minutes`` for n cameras with no open incident, so each run starts its own
    incident; then take what there is (the run then joins the open incident, which still has to
    end up with the right type)."""
    deadline = time.monotonic() + minutes * 60
    while True:
        cameras = api.get("/api/v1/cameras", page_size=200)["items"]
        busy = sorted(c["id"] for c in cameras if c.get("open_incidents"))
        if len(cameras) - len(busy) >= n or time.monotonic() >= deadline:
            if busy:
                print(f"open incidents on {', '.join(busy)}: runs there may join them", flush=True)
            return pick_cameras(cameras, n)
        print(f"waiting for open incidents to resolve on {', '.join(busy)}", flush=True)
        time.sleep(15)


def follow(api: Api, run: Run) -> None:
    """Find the run's incident and refresh what's known about it."""
    # Fusion attaches a run's reports to an incident already open on the camera (opened by
    # background reports, say) instead of opening a second one, so the run's incident can have
    # been suspected before the trigger: look back over the correlation window and beyond.
    since = (run.triggered_at - timedelta(minutes=30)).isoformat()
    page = api.get("/api/v1/incidents", camera_id=run.camera_id, since=since, page_size=50)
    listed = {i["incident_id"]: i for i in page["items"]}
    for incident_id in listed:
        if incident_id not in run.candidates:
            detail = api.get(f"/api/v1/incidents/{incident_id}")
            reports = detail["evidence"]["reports"]
            if not any(r["report"].get("scenario_id") == run.scenario_id for r in reports):
                continue
            run.candidates.append(incident_id)
    details = []
    for incident_id in run.candidates:
        detail = api.get(f"/api/v1/incidents/{incident_id}")
        mine = sum(
            r["report"].get("scenario_id") == run.scenario_id for r in detail["evidence"]["reports"]
        )
        details.append((detail["incident"]["incident_type"] == run.expected_type, mine, detail))
    if not details:
        return
    best = max(details, key=lambda d: (d[0], d[1]))[2]
    inc = best["incident"]
    run.incident_id = inc["incident_id"]
    run.incident_type = inc["incident_type"]
    run.status = inc["status"]
    run.start_time = ts(inc.get("start_time"))
    run.first_evidence_at = ts(inc.get("first_evidence_at"))
    run.suspected_at = ts(inc.get("suspected_at"))
    run.confirmed_at = ts(inc.get("confirmed_at"))
    run.end_time = ts(inc.get("end_time"))
    run.summaries = len(best["summaries"])
    stamps = [ts(s.get("timestamp")) for s in best["summaries"] if s.get("timestamp")]
    run.summary_at = min(stamps) if stamps else None
    run.fetched = True
    run.listed = run.incident_id in listed or run.listed


def refresh_runs(api: Api, runs: list[Run]) -> None:
    for run in runs:
        page = api.get("/api/v1/scenarios/runs", camera_id=run.camera_id, page_size=20)
        for r in page["items"]:
            if str(r["id"]) == run.scenario_id:
                run.run_status = r["status"]
                run.run_started_at = ts(r.get("started_at"))
                run.run_ended_at = ts(r.get("ended_at"))


def table(runs: list[Run]) -> str:
    head = [
        "scenario",
        "camera",
        "expected",
        "got",
        *CHECKS,
        "suspected",
        "confirmed",
        "summary",
        "resolved",
        "start vs run",
        "result",
    ]
    rows = []
    for r in runs:
        res = r.results()
        offset = "-"
        if r.start_time and r.run_started_at:
            offset = f"{(r.start_time - r.run_started_at).total_seconds():+.0f}s"
        rows.append(
            [
                r.scenario,
                r.camera_id.removeprefix("cam-"),
                r.expected_type,
                r.incident_type or "-",
                *("ok" if res[c] else "FAIL" for c in CHECKS),
                r.after(r.suspected_at),
                r.after(r.confirmed_at),
                r.after(r.summary_at),
                r.after(r.end_time),
                offset,
                "PASS" if r.passed else "FAIL",
            ]
        )
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows, strict=True)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=True))  # noqa: E731
    return "\n".join([line(head), line("-" * w for w in widths), *map(line, rows)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--api", default="http://localhost:8001")
    parser.add_argument("--time-scale", type=float, default=6.0)
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--wait-free-minutes",
        type=float,
        default=10.0,
        help="wait this long for cameras without an open incident before triggering",
    )
    parser.add_argument(
        "--prometheus",
        default="",
        help="Prometheus URL: wait for anomaly_service's camera warm-up before triggering",
    )
    parser.add_argument("--scenarios", default="", help="comma-separated; default: all")
    parser.add_argument("--json", default="", help="write the results here")
    args = parser.parse_args()
    api = Api(args.api)

    ready = api.get("/ready")
    if ready.get("status") != "ready":
        print(f"API not ready: {ready}", file=sys.stderr)
        return 2
    scenarios = api.get("/api/v1/scenarios")
    wanted = [s for s in args.scenarios.split(",") if s]
    if wanted:
        scenarios = [s for s in scenarios if s["name"] in wanted]
    scenarios.sort(key=lambda s: s["name"])
    wait_for_warm_up(args.prometheus, args.wait_free_minutes)
    cameras = wait_for_cameras(api, len(scenarios), args.wait_free_minutes)
    wait_for_warm_up(args.prometheus, args.wait_free_minutes)  # a restart while waiting
    runs = [
        Run(s["name"], cam, s["incident_type"], s["duration_seconds"] / args.time_scale)
        for s, cam in zip(scenarios, cameras, strict=True)
    ]

    started = datetime.now(UTC)
    for run in runs:
        resp = api.post(
            f"/api/v1/scenarios/{run.scenario}/trigger",
            {"camera_id": run.camera_id, "time_scale": args.time_scale},
        )
        run.scenario_id, run.triggered_at = str(resp["scenario_id"]), datetime.now(UTC)
        print(
            f"triggered {run.scenario} on {run.camera_id} at x{args.time_scale:g} "
            f"(runs ~{run.duration_s / 60:.1f} min): {run.scenario_id}",
            flush=True,
        )

    deadline = time.monotonic() + args.timeout_minutes * 60
    last_line = ""
    while time.monotonic() < deadline:
        time.sleep(args.poll_seconds)
        try:
            refresh_runs(api, runs)
            for run in runs:
                if not run.done:
                    follow(api, run)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  API error (retrying): {exc}", flush=True)
            continue
        elapsed = (datetime.now(UTC) - started).total_seconds()
        line = " | ".join(
            f"{r.scenario}: run {r.run_status or '?'}, incident {r.status or 'none yet'}"
            + (f" ({r.incident_type})" if r.incident_type else "")
            for r in runs
        )
        if line != last_line:
            print(f"  [{int(elapsed // 60):02d}:{int(elapsed % 60):02d}] {line}", flush=True)
            last_line = line
        if all(r.done for r in runs):
            break

    print()
    print(table(runs))
    total = (datetime.now(UTC) - started).total_seconds()
    passed = sum(r.passed for r in runs)
    print(f"\n{passed}/{len(runs)} scenarios passed in {total / 60:.1f} min", flush=True)
    for r in runs:
        if not r.passed:
            failed = [c for c, ok in r.results().items() if not ok]
            print(
                f"  {r.scenario}: failed {', '.join(failed)}; run {r.run_status or 'not seen'}, "
                f"incident {r.incident_id or 'none'} ({r.status or '-'}), "
                f"candidates {r.candidates or 'none'}"
            )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "passed": passed == len(runs),
                    "seconds": round(total),
                    "time_scale": args.time_scale,
                    "runs": [
                        {**asdict(r), "results": r.results(), "passed": r.passed} for r in runs
                    ],
                },
                f,
                indent=2,
                default=str,
            )
    return 0 if passed == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
