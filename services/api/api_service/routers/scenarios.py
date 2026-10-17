"""Scenarios: what can be run (the event simulator's YAML files), recent runs, and triggering
one by publishing a SimulatorCommand to simulator.control."""

import re
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from api_service.db import connection
from api_service.responses import (
    Page,
    ScenarioInfo,
    ScenarioPhase,
    ScenarioRunRecord,
    TriggerRequest,
    TriggerResponse,
)
from api_service.routers.common import Paging, paging
from traffic_common.schemas import SimulatorCommand
from traffic_common.topics import Topic

router = APIRouter(prefix="/api/v1/scenarios", tags=["scenarios"])
_UNITS = {"s": 1, "m": 60, "h": 3600}


def parse_duration(value) -> float:
    """'3m', '45s', '1h', 90 -> seconds."""
    if isinstance(value, int | float):
        return float(value)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", str(value))
    if not m:
        raise ValueError(f"bad duration {value!r}")
    return float(m.group(1)) * _UNITS[m.group(2) or "s"]


def load_scenarios(directory: Path) -> dict[str, ScenarioInfo]:
    out = {}
    for path in sorted(directory.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        phases = [
            ScenarioPhase(
                name=p["name"],
                duration_seconds=parse_duration(p["duration"]),
                severity=p.get("severity", "low"),
            )
            for p in spec.get("phases", [])
        ]
        out[path.stem] = ScenarioInfo(
            name=spec.get("name", path.stem),
            description=" ".join(str(spec.get("description", "")).split()),
            incident_type=spec.get("incident_type", "none"),
            duration_seconds=sum(p.duration_seconds for p in phases),
            phases=phases,
        )
    return out


@router.get("", response_model=list[ScenarioInfo])
async def list_scenarios(request: Request) -> list[ScenarioInfo]:
    return list(load_scenarios(request.app.state.config.scenarios_dir).values())


@router.get("/runs", response_model=Page[ScenarioRunRecord])
async def list_runs(
    camera_id: str | None = None,
    conn: AsyncConnection = Depends(connection),
    p: Paging = Depends(paging),
) -> Page[ScenarioRunRecord]:
    where = "WHERE camera_id = :c" if camera_id else ""
    params = {"c": camera_id, "limit": p.page_size, "offset": p.offset}
    total = await conn.scalar(text(f"SELECT count(*) FROM scenario_runs {where}"), params)
    rows = await conn.execute(
        text(
            "SELECT id, scenario_name, camera_id, incident_type, peak_severity, status, "
            "triggered_by, time_scale, started_at, expected_end_at, ended_at, reports_emitted, "
            f"events_emitted, timeline, error FROM scenario_runs {where} "
            "ORDER BY started_at DESC LIMIT :limit OFFSET :offset"
        ),
        params,
    )
    return Page(
        items=[ScenarioRunRecord(**r) for r in rows.mappings()],
        total=total,
        page=p.page,
        page_size=p.page_size,
    )


@router.post("/{scenario}/trigger", response_model=TriggerResponse, status_code=202)
async def trigger(
    scenario: str,
    body: TriggerRequest,
    request: Request,
    conn: AsyncConnection = Depends(connection),
) -> TriggerResponse:
    available = load_scenarios(request.app.state.config.scenarios_dir)
    if scenario not in available:
        raise HTTPException(404, f"unknown scenario {scenario!r}; available: {sorted(available)}")
    exists = await conn.scalar(text("SELECT 1 FROM cameras WHERE id = :c"), {"c": body.camera_id})
    if not exists:
        raise HTTPException(404, f"unknown camera {body.camera_id!r}")
    command = SimulatorCommand(
        action="start_scenario",
        scenario=scenario,
        camera_id=body.camera_id,
        time_scale=body.time_scale,
        requested_by="api",
    )
    producer = request.app.state.producer
    producer.send(Topic.SIMULATOR_CONTROL, command)
    remaining = await run_in_threadpool(producer.flush, 5.0)
    if remaining:
        raise HTTPException(503, "could not deliver the command to Kafka")
    return TriggerResponse(
        scenario_id=command.scenario_id,
        scenario=scenario,
        camera_id=body.camera_id,
        time_scale=body.time_scale,
    )
