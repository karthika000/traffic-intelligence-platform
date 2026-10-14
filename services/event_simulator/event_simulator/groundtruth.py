"""Ground truth in Postgres: every emitted report (with labels) and every scenario run."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, insert, update

from event_simulator.reports import GeneratedReport
from event_simulator.scenarios import Scenario
from traffic_common.db.models import ScenarioRun, SimulatedReport
from traffic_common.enums import ScenarioStatus


class GroundTruthStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def record_report(
        self,
        report_id: uuid.UUID,
        report: GeneratedReport,
        emitted_at: datetime,
        scenario_id: uuid.UUID | None = None,
        phase: str | None = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                insert(SimulatedReport).values(
                    id=report_id,
                    scenario_run_id=scenario_id,
                    camera_id=report.camera_id,
                    source=report.source,
                    style=report.style.value,
                    text=report.text,
                    incident_type=report.incident_type,
                    severity=report.severity,
                    location_mention=report.location_mention or None,
                    stage=report.stage.value,
                    phase=phase,
                    emitted_at=emitted_at,
                )
            )

    def start_run(
        self,
        run_id: uuid.UUID,
        scenario: Scenario,
        camera_id: str,
        triggered_by: str,
        time_scale: float,
        started_at: datetime,
        expected_end_at: datetime,
        timeline: list[dict[str, Any]],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                insert(ScenarioRun).values(
                    id=run_id,
                    scenario_name=scenario.name,
                    camera_id=camera_id,
                    incident_type=scenario.incident_type,
                    peak_severity=scenario.peak_severity,
                    status=ScenarioStatus.RUNNING,
                    triggered_by=triggered_by,
                    time_scale=time_scale,
                    started_at=started_at,
                    expected_end_at=expected_end_at,
                    timeline=timeline,
                    definition=scenario.model_dump(mode="json"),
                )
            )

    def finish_run(
        self,
        run_id: uuid.UUID,
        status: ScenarioStatus,
        ended_at: datetime,
        reports_emitted: int,
        events_emitted: int,
        error: str | None = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(ScenarioRun)
                .where(ScenarioRun.id == run_id)
                .values(
                    status=status,
                    ended_at=ended_at,
                    reports_emitted=reports_emitted,
                    events_emitted=events_emitted,
                    error=error,
                )
            )
