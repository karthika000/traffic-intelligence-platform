import { useMemo } from "react";
import { useThrottledEffect } from "../lib/hooks";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { type IncidentFilters, useCameras, useIncidents } from "../api/queries";
import { ErrorNote, Field, Pager, Panel, ScoreBar, SeverityChip, Skeleton, StatusChip } from "../components/ui";
import { dateTime, titleCase } from "../lib/format";
import { headline, startedAt } from "../lib/incidents";
import { SEVERITIES } from "../lib/scales";
import { useLive } from "../store/live";

const PAGE_SIZE = 25;
const TYPES = ["accident", "breakdown", "road_works", "congestion", "weather_hazard", "debris", "police_activity", "none"];
const STATUSES = ["suspected", "confirmed", "resolved"];
const RANGES: Record<string, number | null> = { "1h": 1, "6h": 6, "24h": 24, "7d": 168, all: null };


export function Incidents() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const cameras = useCameras();
  const names = useMemo(() => Object.fromEntries((cameras.data ?? []).map((c) => [c.id, c.name])), [cameras.data]);

  const range = params.get("range") ?? "24h";
  const page = Number(params.get("page") ?? 1);
  const filters: IncidentFilters = useMemo(() => {
    const hours = RANGES[range] ?? null;
    return {
      status: params.get("status") ?? undefined,
      camera_id: params.get("camera") ?? undefined,
      incident_type: params.get("type") ?? undefined,
      severity: params.get("severity") ?? undefined,
      // Rounded to the minute so the query key (and cache) is stable between renders.
      since: hours ? new Date(Math.floor(Date.now() / 60_000) * 60_000 - hours * 3_600_000).toISOString() : undefined,
      page,
      page_size: PAGE_SIZE,
    };
  }, [params, range, page]);
  const q = useIncidents(filters);
  const summaries = useLive((s) => s.summaries);
  const live = useLive((s) => s.incidents);
  const tick = useLive((s) => s.ticks.incidents);
  // New versions arrive live: refetch the page, at most every 3 s.
  useThrottledEffect(tick, 3000, () => void q.refetch());

  const set = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    if (key !== "page") next.delete("page");
    setParams(next, { replace: true });
  };

  const select = (key: string, label: string, options: { value: string; label: string }[], allLabel = "All") => (
    <Field label={label}>
      <select className="input" value={params.get(key) ?? ""} onChange={(e) => set(key, e.target.value)} aria-label={label}>
        <option value="">{allLabel}</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-3">
        {select("status", "Status", STATUSES.map((s) => ({ value: s, label: titleCase(s) })))}
        {select("type", "Type", TYPES.map((s) => ({ value: s, label: titleCase(s) })))}
        {select("severity", "Severity", SEVERITIES.map((s) => ({ value: s, label: titleCase(s) })))}
        {select("camera", "Camera", (cameras.data ?? []).map((c) => ({ value: c.id, label: c.name })))}
        <Field label="Started within">
          <select className="input" value={range} onChange={(e) => set("range", e.target.value)} aria-label="Time range">
            {Object.keys(RANGES).map((k) => (
              <option key={k} value={k}>
                {k === "all" ? "Any time" : `Last ${k}`}
              </option>
            ))}
          </select>
        </Field>
        {[...params.keys()].some((k) => k !== "page") && (
          <button className="btn" onClick={() => setParams({}, { replace: true })}>
            Reset
          </button>
        )}
        {q.isFetching && <span className="text-xs text-faint pb-2">Updating…</span>}
      </div>

      {q.isError && <ErrorNote error={q.error} onRetry={() => q.refetch()} />}

      <Panel bodyClass="p-0 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-line">
            <tr>
              <th className="th">Started</th>
              <th className="th">Incident</th>
              <th className="th">Camera</th>
              <th className="th">Type</th>
              <th className="th">Severity</th>
              <th className="th">Status</th>
              <th className="th">Confidence</th>
              <th className="th">Ver.</th>
            </tr>
          </thead>
          <tbody>
            {q.data
              ? q.data.items.map((rest) => {
                  // Prefer a newer live version of the row if one has arrived since the fetch.
                  const l = live[rest.incident_id];
                  const i = l && l.version > rest.version ? l : rest;
                  return (
                    <tr
                      key={i.incident_id}
                      className="border-b border-line last:border-0 hover:bg-raised cursor-pointer"
                      onClick={() => navigate(`/incidents/${i.incident_id}`)}
                    >
                      <td className="td whitespace-nowrap tabular-nums text-muted">{dateTime(startedAt(i))}</td>
                      <td className="td">
                        <Link to={`/incidents/${i.incident_id}`} className="font-medium hover:underline" onClick={(e) => e.stopPropagation()}>
                          {headline(i, summaries[i.incident_id])}
                        </Link>
                      </td>
                      <td className="td text-muted whitespace-nowrap">{names[i.camera_id] ?? i.camera_id}</td>
                      <td className="td whitespace-nowrap">{titleCase(i.incident_type)}</td>
                      <td className="td"><SeverityChip severity={i.severity} /></td>
                      <td className="td"><StatusChip status={i.status} /></td>
                      <td className="td"><ScoreBar value={i.fused_confidence} /></td>
                      <td className="td tabular-nums text-muted">v{i.version}</td>
                    </tr>
                  );
                })
              : [0, 1, 2, 3, 4].map((n) => (
                  <tr key={n}>
                    <td className="td" colSpan={8}><Skeleton className="h-5" /></td>
                  </tr>
                ))}
          </tbody>
        </table>
        {q.data && q.data.items.length === 0 && <div className="py-8 text-center text-sm text-faint">No incidents match these filters.</div>}
        {q.data && <Pager page={page} pageSize={PAGE_SIZE} total={q.data.total} onPage={(p) => set("page", String(p))} />}
      </Panel>
    </div>
  );
}
