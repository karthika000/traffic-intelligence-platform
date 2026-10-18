import { useEffect, useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { keys, useCameras, useIncident } from "../api/queries";
import type { Incident, IncidentDetail as Detail, IncidentVersion, Summary } from "../api/types";
import { CongestionChip, Empty, ErrorNote, Panel, ScoreBar, SeverityChip, Skeleton, StatusChip, Tag } from "../components/ui";
import { dateTime, num, pct, shortId, titleCase } from "../lib/format";
import { startedAt } from "../lib/incidents";
import { severityStyle } from "../lib/scales";
import { useTheme } from "../lib/theme";
import { type Components, confidenceParts } from "../lib/confidence";
import { SERIES } from "../lib/scales";
import { demoStore, evidenceFromRun, useDemo } from "../store/demo";
import { liveStore, useLive } from "../store/live";


function ConfidenceBreakdown({ incident }: { incident: Incident }) {
  const theme = useTheme();
  const c = (incident.confidence_components ?? {}) as Components;
  const parts = confidenceParts(c);
  const colors = { vision: SERIES[theme][0], text: SERIES[theme][1], agreement: "#8b8a84" };
  const rows = (["vision", "text", "agreement"] as const).map((k) => ({ key: k, ...parts[k] }));
  const degraded = incident.vision_degraded || c.vision?.degraded;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-baseline gap-2">
        <span className="text-3xl font-semibold tabular-nums">{pct(incident.fused_confidence)}</span>
        <span className="text-xs text-muted">fused confidence · {parts.primary} led</span>
      </div>
      <div className="flex h-4 w-full rounded overflow-hidden bg-line gap-[2px]" role="img" aria-label="Confidence contributions">
        {rows.map((r) =>
          r.contribution > 0 ? (
            <div key={r.key} style={{ width: `${r.contribution * 100}%`, background: colors[r.key] }} title={`${r.key}: ${num(r.contribution, 3)}`} />
          ) : null,
        )}
      </div>
      <table className="text-xs w-full">
        <thead>
          <tr className="text-faint">
            <th className="text-left font-medium py-1">Signal</th>
            <th className="text-left font-medium">Score</th>
            <th className="text-right font-medium">Weight</th>
            <th className="text-right font-medium">Contribution</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className="border-t border-line">
              <td className="py-1.5">
                <span className="inline-flex items-center gap-1.5 capitalize">
                  <span className="w-2.5 h-2.5 rounded-sm" style={{ background: colors[r.key] }} />
                  {r.key}
                </span>
              </td>
              <td><ScoreBar value={r.score} color={colors[r.key]} /></td>
              <td className="text-right tabular-nums">{num(r.weight, 2)}</td>
              <td className="text-right tabular-nums font-medium">{num(r.contribution, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {degraded && (
        <div className="rounded-md border border-[#fab219]/50 bg-[#fab219]/10 px-2.5 py-2 text-xs">
          <strong>Vision was degraded.</strong> The camera evidence was down-weighted
          {c.vision?.raw_score !== undefined && c.vision?.score !== undefined && c.vision.raw_score.toFixed(2) !== c.vision.score.toFixed(2) &&
            ` (score ${num(c.vision.raw_score, 2)} → ${num(c.vision.score, 2)})`}
          {c.vision?.hints?.includes("camera_degraded") && "; the camera reported itself degraded"}, so reports carry more of the decision.
        </div>
      )}
      {c.severity && (
        <div className="flex flex-wrap gap-2 text-xs text-muted items-center">
          Severity inputs:
          {c.severity.text && <span>text <SeverityChip severity={c.severity.text} /></span>}
          {c.severity.vision && <span>vision <SeverityChip severity={c.severity.vision} /></span>}
          {c.severity.congestion && <span>congestion <CongestionChip level={c.severity.congestion} /></span>}
        </div>
      )}
    </div>
  );
}

function Timeline({ versions, current }: { versions: IncidentVersion[]; current: Incident }) {
  const list = versions.length ? versions : [{ version: current.version, status: current.status, severity: current.severity, incident_type: current.incident_type, fused: current.fused_confidence, reasons: current.change_reasons }];
  return (
    <ol className="relative ml-2 border-l border-line">
      {list.map((v) => (
        <li key={v.version} className="ml-4 pb-4 last:pb-0">
          <span className="absolute -left-[7px] mt-1 w-3.5 h-3.5 rounded-full border-2 border-panel" style={{ background: severityStyle(v.severity).color }} />
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold tabular-nums">v{v.version}</span>
            <StatusChip status={v.status} />
            <SeverityChip severity={v.severity} />
            <Tag>{titleCase(v.incident_type)}</Tag>
            {v.fused !== undefined && <span className="text-xs text-muted tabular-nums">conf {pct(v.fused)}</span>}
          </div>
          {!!v.reasons?.length && <div className="mt-1 text-xs text-muted">{v.reasons.join(" · ")}</div>}
        </li>
      ))}
    </ol>
  );
}

function SummaryCard({ s, latest }: { s: Summary; latest?: boolean }) {
  const f = (s.faithfulness ?? {}) as { passed?: boolean; fallback_reason?: string; triggers?: string[]; configuration?: string };
  return (
    <div className={`rounded-md border p-3 ${latest ? "border-accent/60 bg-accent/5" : "border-line"}`}>
      <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
        <Tag>written for v{s.incident_version ?? "?"}</Tag>
        <Tag title="generation path">{s.generation ?? "unknown"}</Tag>
        <span>
          {s.model_name}@{s.model_version}
          {s.prompt_version ? ` · prompt ${s.prompt_version}` : ""}
        </span>
        {f.passed !== undefined && <span className={f.passed ? "text-[#0ca30c]" : "text-[#ee4b26]"}>{f.passed ? "✓ faithfulness passed" : "✗ faithfulness failed"}</span>}
        <span className="ml-auto tabular-nums">{dateTime(s.timestamp)}</span>
      </div>
      {s.headline && <div className="mt-1.5 font-medium text-sm">{s.headline}</div>}
      <p className="mt-1 text-sm text-muted leading-relaxed">{s.text}</p>
      {(f.fallback_reason || f.triggers?.length) && (
        <div className="mt-1.5 text-[11px] text-faint">
          {f.triggers?.length ? `triggers: ${f.triggers.join(", ")}` : ""}
          {f.fallback_reason ? ` · fallback: ${f.fallback_reason}` : ""}
        </div>
      )}
    </div>
  );
}

function Evidence({ detail }: { detail: Detail }) {
  const { anomalies, reports, metrics, weights = {} } = detail.evidence;
  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
      <Panel title={`Reports (${reports.length})`} bodyClass="p-0 max-h-[460px] overflow-y-auto" className="xl:col-span-2">
        {reports.length ? (
          <ul>
            {reports.map(({ report: r, nlp, unlocated }) => (
              <li key={r.report_id} className="px-3 py-2.5 border-b border-line last:border-0">
                <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
                  <Tag>{r.source}</Tag>
                  <span className="tabular-nums">{dateTime(r.timestamp)}</span>
                  {r.scenario_id && <Tag title={r.scenario_id}>scenario {shortId(r.scenario_id)}</Tag>}
                  {weights[`report:${r.report_id}`] != null && <span>weight {num(weights[`report:${r.report_id}`], 2)}</span>}
                </div>
                <p className="mt-1 text-sm">{r.text}</p>
                {nlp ? (
                  <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
                    <span>
                      <span className="text-ink font-medium">{titleCase(nlp.incident_type)}</span> {pct(nlp.incident_confidence)}
                    </span>
                    {nlp.severity && (
                      <span className="inline-flex items-center gap-1">
                        <SeverityChip severity={nlp.severity} /> {pct(nlp.severity_confidence)}
                      </span>
                    )}
                    <span>{unlocated ? "unlocated" : `→ ${nlp.camera_id} ${pct(nlp.camera_confidence)}`}</span>
                    {nlp.stage && <span>stage {nlp.stage}</span>}
                  </div>
                ) : (
                  <div className="mt-1 text-xs text-faint">No NLP result yet.</div>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <Empty>No reports attached.</Empty>
        )}
      </Panel>
      <Panel title={`Anomalies (${anomalies.length})`} bodyClass="p-0">
        {anomalies.length ? (
          <ul>
            {anomalies.map(({ anomaly: a, status, detected_at }) => (
              <li key={a.anomaly_id} className="px-3 py-2 border-b border-line last:border-0 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{titleCase(a.anomaly_type)}</span>
                  {a.hints.map((h) => <Tag key={h}>{titleCase(h)}</Tag>)}
                  <span className="ml-auto text-[11px] text-faint">{dateTime(detected_at)}</span>
                </div>
                <div className="text-xs text-muted mt-1 tabular-nums">
                  {a.scope === "system"
                    ? `${num(a.score)} cameras degraded (minimum ${num(a.threshold)})`
                    : `score ${num(a.score, 2)} / threshold ${num(a.threshold, 2)}`}{" "}
                  · {a.metric_name} {num(a.observed_value, 2)} · {status} · {a.model_name}@{a.model_version}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <Empty>No anomalies attached.</Empty>
        )}
      </Panel>
      <Panel title={`Metric windows (${metrics.length})`} bodyClass="p-0 overflow-x-auto">
        {metrics.length ? (
          <table className="w-full text-xs tabular-nums">
            <thead className="border-b border-line">
              <tr>
                <th className="th">Window end</th>
                <th className="th">Vehicles</th>
                <th className="th">Avg km/h</th>
                <th className="th">p85</th>
                <th className="th">Density</th>
                <th className="th">Stopped</th>
                <th className="th">Peds</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => (
                <tr key={m.message_id} className="border-b border-line last:border-0">
                  <td className="td">{dateTime(m.window_end)}</td>
                  <td className="td">{m.vehicle_count}</td>
                  <td className="td">{num(m.avg_speed_kmh, 1)}</td>
                  <td className="td">{num(m.p85_speed_kmh, 1)}</td>
                  <td className="td">{num(m.density, 2)}</td>
                  <td className="td">{m.stopped_vehicles}</td>
                  <td className="td">{m.pedestrians_in_roadway}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Empty>No metric windows attached.</Empty>
        )}
      </Panel>
    </div>
  );
}

export function IncidentDetail() {
  const { id = "" } = useParams();
  const q = useIncident(id);
  const qc = useQueryClient();
  const cameras = useCameras();
  const liveVersion = useLive((s) => s.incidents[id]?.version);
  const liveSummary = useLive((s) => s.summaries[id]?.summary_id);
  const runs = useDemo((s) => s.runs);

  // A newer version or summary arrived live: refetch the detail (evidence and history change).
  useEffect(() => {
    if (!q.data) return;
    const known = q.data.summaries.some((s) => s.summary_id === liveSummary);
    if ((liveVersion && liveVersion > q.data.incident.version) || (liveSummary && !known)) {
      void qc.invalidateQueries({ queryKey: keys.incident(id) });
    }
  }, [liveVersion, liveSummary, q.data, qc, id]);

  useEffect(() => {
    if (!q.data) return;
    liveStore.getState().seedIncidents([q.data.incident]);
    liveStore.getState().seedSummaries(q.data.summaries);
    for (const r of demoStore.getState().runs)
      if (!r.linked && r.cameraId === q.data.incident.camera_id && evidenceFromRun(q.data, r.scenarioId))
        demoStore.getState().confirm(r.scenarioId, q.data.incident.incident_id, true);
  }, [q.data]);

  const summaries = useMemo(
    () => [...(q.data?.summaries ?? [])].sort((a, b) => (a.incident_version ?? 0) - (b.incident_version ?? 0) || (a.timestamp < b.timestamp ? -1 : 1)),
    [q.data?.summaries],
  );

  if (q.isError && !q.data) {
    return (
      <div className="flex flex-col gap-3">
        <Link to="/incidents" className="text-sm link">← Incidents</Link>
        <ErrorNote error={q.error} onRetry={() => q.refetch()} />
      </div>
    );
  }
  if (!q.data) return <Skeleton className="h-96" />;

  const d = q.data;
  const i = d.incident;
  const latest = summaries[summaries.length - 1];
  const cam = cameras.data?.find((c) => c.id === i.camera_id);
  const demo = runs.find((r) => r.incidentId === i.incident_id);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <Link to="/incidents" className="text-sm link">← Incidents</Link>
        <span className="text-faint">/</span>
        <code className="text-xs text-muted">{shortId(i.incident_id)}</code>
        {demo && (
          <span className="text-xs font-semibold text-accent ring-1 ring-accent rounded px-1.5 py-0.5">
            ◆ produced by demo {demo.scenario} {demo.linked ? "(linked by its reports)" : "(probable)"}
          </span>
        )}
        {q.isFetching && <span className="text-xs text-faint">Updating…</span>}
      </div>

      <Panel>
        <div className="flex flex-wrap items-start gap-4">
          <div className="flex-1 min-w-[280px]">
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <SeverityChip severity={i.severity} />
              <StatusChip status={i.status} />
              <Tag>{titleCase(i.incident_type)}</Tag>
              <Tag>v{i.version}</Tag>
              <Link to={`/cameras/${i.camera_id}`} className="text-xs link">{cam?.name ?? i.camera_id}</Link>
            </div>
            <h1 className="text-xl font-semibold leading-snug">{latest?.headline || i.title}</h1>
            {latest ? (
              <>
                <p className="mt-2 text-sm leading-relaxed">{latest.text}</p>
                <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted">
                  <Tag title="how this summary was generated">generation: {latest.generation ?? "unknown"}</Tag>
                  <span>{latest.model_name}@{latest.model_version}</span>
                  <span>written for v{latest.incident_version}{latest.incident_version !== i.version ? ` (incident is now v${i.version})` : ""}</span>
                  {latest.latency_ms != null && <span>{num(latest.latency_ms, 1)} ms</span>}
                </div>
              </>
            ) : (
              <p className="mt-2 text-sm text-muted">{i.description ?? "No summary yet."}</p>
            )}
          </div>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs min-w-[240px]">
            <dt className="text-faint" title="first evidence supporting the incident's type">Started</dt><dd className="tabular-nums">{dateTime(startedAt(i), true)}</dd>
            {i.first_evidence_at && i.first_evidence_at !== i.start_time && (
              <>
                <dt className="text-faint" title="earliest evidence attached, of any type (can predate the incident)">First evidence</dt>
                <dd className="tabular-nums">{dateTime(i.first_evidence_at, true)}</dd>
              </>
            )}
            <dt className="text-faint">Suspected</dt><dd className="tabular-nums">{dateTime(i.suspected_at, true)}</dd>
            <dt className="text-faint">Confirmed</dt><dd className="tabular-nums">{dateTime(i.confirmed_at, true)}</dd>
            <dt className="text-faint">Ended</dt><dd className="tabular-nums">{i.end_time ? dateTime(i.end_time, true) : "–"}</dd>
            {d.resolution && (<><dt className="text-faint">Resolution</dt><dd>{d.resolution}</dd></>)}
          </dl>
        </div>
      </Panel>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title="Status timeline">
          <Timeline versions={d.versions} current={i} />
        </Panel>
        <Panel title="Confidence breakdown">
          <ConfidenceBreakdown incident={i} />
        </Panel>
      </div>

      <Evidence detail={d} />

      <Panel title={`Summaries (${summaries.length})`}>
        {summaries.length ? (
          <div className="flex flex-col gap-2">
            {[...summaries].reverse().map((s) => (
              <SummaryCard key={s.summary_id} s={s} latest={s === latest} />
            ))}
          </div>
        ) : (
          <Empty>No summaries written yet.</Empty>
        )}
      </Panel>
    </div>
  );
}
