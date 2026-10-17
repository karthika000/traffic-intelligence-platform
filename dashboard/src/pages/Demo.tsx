import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useShallow } from "zustand/react/shallow";
import { useCameras, useScenarioRuns, useScenarios, useTriggerScenario } from "../api/queries";
import type { ScenarioRun } from "../api/types";
import { IncidentFeed } from "../components/IncidentFeed";
import { ScenarioTriggerForm } from "../components/ScenarioTriggerForm";
import { Empty, ErrorNote, Panel, SeverityChip, StatusChip, Tag } from "../components/ui";
import { dateTime, shortId } from "../lib/format";
import { feed, headline } from "../lib/incidents";
import { demoIncidentIds, demoStore, useDemo } from "../store/demo";
import { useLive } from "../store/live";

function Progress({ run, now }: { run: ScenarioRun; now: number }) {
  const start = new Date(run.started_at).getTime();
  const end = new Date(run.expected_end_at).getTime();
  const p = Math.max(0, Math.min(1, (now - start) / Math.max(1, end - start)));
  const phase = run.timeline.find((t) => {
    const s = new Date(String(t.starts_at)).getTime();
    const e = new Date(String(t.ends_at)).getTime();
    return now >= s && now < e;
  });
  return (
    <div className="min-w-[140px]">
      <div className="h-1.5 rounded-full bg-line overflow-hidden">
        <div className="h-full bg-accent transition-[width] duration-1000" style={{ width: `${p * 100}%` }} />
      </div>
      <div className="mt-1 text-[11px] text-muted">{phase ? String(phase.phase) : "finishing"} · {Math.round(p * 100)}%</div>
    </div>
  );
}

export function Demo() {
  const scenarios = useScenarios();
  const cameras = useCameras();
  const tracked = useDemo(useShallow((s) => s.runs));
  const anyRunning = tracked.some((r) => Date.now() - r.triggeredAt < 30 * 60_000);
  const runs = useScenarioRuns(anyRunning ? 3000 : 10_000);
  const trigger = useTriggerScenario();
  const incidents = useLive((s) => s.incidents);
  const summaries = useLive((s) => s.summaries);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  const names = useMemo(() => Object.fromEntries((cameras.data ?? []).map((c) => [c.id, c.name])), [cameras.data]);
  const demoMap = useMemo(() => demoIncidentIds(tracked), [tracked]);
  const produced = useMemo(
    () => feed(Object.fromEntries(Object.entries(incidents).filter(([id]) => demoMap.has(id))), 10),
    [incidents, demoMap],
  );
  const running = (runs.data ?? []).filter((r) => r.status === "running");

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <div className="xl:col-span-2 flex flex-col gap-4">
        <Panel title="Trigger a scenario">
          {scenarios.isError && <ErrorNote error={scenarios.error} onRetry={() => scenarios.refetch()} />}
          <ScenarioTriggerForm
            scenarios={scenarios.data ?? []}
            cameras={cameras.data ?? []}
            defaults={{ scenario: "multi_vehicle_accident", camera_id: "cam-hyde-park-corner", time_scale: 6 }}
            onTrigger={async ({ scenario, camera_id, time_scale }) => {
              const triggeredAt = Date.now();
              const res = await trigger.mutateAsync({ scenario, camera_id, time_scale });
              demoStore.getState().track({ scenarioId: res.scenario_id, scenario, cameraId: camera_id, timeScale: time_scale, triggeredAt });
              return res;
            }}
          />
        </Panel>

        <Panel
          title="Triggered from this browser"
          actions={
            tracked.length > 0 && (
              <button className="text-xs link" onClick={() => demoStore.getState().clear()}>
                Clear
              </button>
            )
          }
          bodyClass="p-0"
        >
          {tracked.length === 0 ? (
            <Empty>Runs you trigger here are tracked until they produce an incident.</Empty>
          ) : (
            <ul>
              {tracked.map((r) => {
                const inc = r.incidentId ? incidents[r.incidentId] : undefined;
                const run = runs.data?.find((x) => x.id === r.scenarioId);
                return (
                  <li key={r.scenarioId} className="flex flex-wrap items-center gap-3 px-3 py-2.5 border-b border-line last:border-0">
                    <div className="min-w-[220px]">
                      <div className="text-sm font-medium">{r.scenario}</div>
                      <div className="text-[11px] text-muted">
                        {names[r.cameraId] ?? r.cameraId} · ×{r.timeScale} · {dateTime(new Date(r.triggeredAt).toISOString())} · {shortId(r.scenarioId)}
                      </div>
                    </div>
                    {run && <Tag>{run.status}</Tag>}
                    {run?.status === "running" && <Progress run={run} now={now} />}
                    <div className="ml-auto flex items-center gap-2 min-w-0">
                      {inc ? (
                        <Link to={`/incidents/${inc.incident_id}`} className="flex flex-wrap items-center gap-2 rounded-md ring-2 ring-accent px-2 py-1 bg-accent/10 hover:bg-accent/20" data-testid="produced-incident">
                          <SeverityChip severity={inc.severity} />
                          <StatusChip status={inc.status} />
                          <span className="text-sm font-medium truncate max-w-[340px]">{headline(inc, summaries[inc.incident_id])}</span>
                          <span className="text-[11px] text-muted">v{inc.version}{r.linked ? " · linked" : " · probable"}</span>
                        </Link>
                      ) : (
                        <span className="text-xs text-muted">Waiting for the incident…</span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </Panel>

        <Panel title={`Scenario runs${running.length ? ` · ${running.length} running` : ""}`} bodyClass="p-0 overflow-x-auto">
          {runs.isError && <div className="p-3"><ErrorNote error={runs.error} onRetry={() => runs.refetch()} /></div>}
          <table className="w-full text-sm">
            <thead className="border-b border-line">
              <tr>
                <th className="th">Scenario</th>
                <th className="th">Camera</th>
                <th className="th">Status</th>
                <th className="th">Started</th>
                <th className="th">Scale</th>
                <th className="th">Peak</th>
                <th className="th">Reports / events</th>
                <th className="th">By</th>
              </tr>
            </thead>
            <tbody>
              {(runs.data ?? []).map((r) => (
                <tr key={r.id} className={`border-b border-line last:border-0 ${demoMap.size && tracked.some((t) => t.scenarioId === r.id) ? "bg-accent/5" : ""}`}>
                  <td className="td font-medium">{r.scenario_name}</td>
                  <td className="td text-muted">{names[r.camera_id] ?? r.camera_id}</td>
                  <td className="td">{r.status === "running" ? <Progress run={r} now={now} /> : <Tag>{r.status}</Tag>}</td>
                  <td className="td whitespace-nowrap tabular-nums">{dateTime(r.started_at)}</td>
                  <td className="td tabular-nums">×{r.time_scale}</td>
                  <td className="td"><SeverityChip severity={r.peak_severity} /></td>
                  <td className="td tabular-nums">{r.reports_emitted} / {r.events_emitted}</td>
                  <td className="td text-muted">{r.triggered_by}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {runs.data?.length === 0 && <Empty>No scenario runs yet.</Empty>}
        </Panel>
      </div>

      <Panel title="Incidents from demo runs" bodyClass="p-0 overflow-y-auto max-h-[70vh]">
        <IncidentFeed incidents={produced} summaries={summaries} cameraNames={names} demoByIncident={demoMap} emptyText="The incident a triggered scenario produces appears here, highlighted." />
      </Panel>
    </div>
  );
}
