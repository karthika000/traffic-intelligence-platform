import { memo, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { type ReportFilters, useCameras, useReports } from "../api/queries";
import type { ReportRecord } from "../api/types";
import { Empty, ErrorNote, Field, Pager, Panel, ScoreBar, SeverityChip, Skeleton, Tag } from "../components/ui";
import { dateTime, titleCase } from "../lib/format";
import { useLive } from "../store/live";
import { useThrottledEffect } from "../lib/hooks";

const TYPES = ["accident", "breakdown", "road_works", "congestion", "weather_hazard", "debris", "police_activity", "none"];
const PAGE_SIZE = 50;

const ReportRow = memo(function ReportRow({ rec, fresh, cameraName }: { rec: ReportRecord; fresh: boolean; cameraName?: string }) {
  const { report: r, nlp, unlocated } = rec;
  return (
    <tr className={`border-b border-line last:border-0 ${fresh ? "animate-flash" : ""}`} data-testid="report-row">
      <td className="td whitespace-nowrap tabular-nums text-muted">{dateTime(r.timestamp)}</td>
      <td className="td"><Tag>{r.source}</Tag></td>
      <td className="td max-w-[520px]">
        <div className="text-sm">{r.text}</div>
        {r.scenario_id && <span className="text-[11px] text-accent">◆ scenario {r.scenario_id.slice(0, 8)}</span>}
      </td>
      <td className="td whitespace-nowrap">
        {nlp ? (
          <div className="flex flex-col gap-1">
            <span className="font-medium">{titleCase(nlp.incident_type)}</span>
            <ScoreBar value={nlp.incident_confidence} />
          </div>
        ) : (
          <span className="text-faint text-xs">pending</span>
        )}
      </td>
      <td className="td whitespace-nowrap">
        {nlp?.severity ? (
          <div className="flex flex-col gap-1">
            <SeverityChip severity={nlp.severity} />
            <ScoreBar value={nlp.severity_confidence} />
          </div>
        ) : (
          <span className="text-faint">–</span>
        )}
      </td>
      <td className="td whitespace-nowrap">
        {unlocated || !nlp?.camera_id ? (
          <span className="text-xs text-[#fab219]" title={nlp?.location_name ?? undefined}>
            unlocated{nlp?.location_name ? ` (“${nlp.location_name}”)` : ""}
          </span>
        ) : (
          <div className="flex flex-col gap-1">
            <span className="text-xs">{cameraName ?? nlp.camera_id}</span>
            <ScoreBar value={nlp.camera_confidence} />
          </div>
        )}
      </td>
    </tr>
  );
});

export function Reports() {
  const [params, setParams] = useSearchParams();
  const cameras = useCameras();
  const names = useMemo(() => Object.fromEntries((cameras.data ?? []).map((c) => [c.id, c.name])), [cameras.data]);
  const camera = params.get("camera") ?? "";
  const page = Number(params.get("page") ?? 1);
  const filters: ReportFilters = {
    incident_type: params.get("type") ?? undefined,
    camera_id: camera && camera !== "unlocated" ? camera : undefined,
    unlocated: camera === "unlocated" ? true : undefined,
    page,
    page_size: PAGE_SIZE,
  };
  const q = useReports(filters);
  // Live NLP results mean new reports: refetch page 1, at most every 2 s.
  const tick = useLive((s) => s.ticks.nlp);
  const liveCount = useLive((s) => s.nlp.length);
  useThrottledEffect(tick, 2000, () => {
    if (page === 1) void q.refetch();
  });

  // Flash rows that weren't in the previous fetch.
  const seen = useRef<Set<string> | null>(null);
  const [fresh, setFresh] = useState<Set<string>>(new Set());
  useEffect(() => {
    if (!q.data) return;
    const ids = q.data.items.map((r) => r.report.report_id);
    if (seen.current) setFresh(new Set(ids.filter((id) => !seen.current!.has(id))));
    seen.current = new Set(ids);
  }, [q.data]);

  const set = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    if (key !== "page") next.delete("page");
    seen.current = null;
    setParams(next, { replace: true });
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-3">
        <Field label="Type (NLP)">
          <select className="input" value={params.get("type") ?? ""} onChange={(e) => set("type", e.target.value)} aria-label="Type">
            <option value="">All</option>
            {TYPES.map((t) => (
              <option key={t} value={t}>{titleCase(t)}</option>
            ))}
          </select>
        </Field>
        <Field label="Matched camera">
          <select className="input" value={camera} onChange={(e) => set("camera", e.target.value)} aria-label="Camera">
            <option value="">All, including unlocated</option>
            <option value="unlocated">Unlocated only</option>
            {(cameras.data ?? []).map((c) => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
          </select>
        </Field>
        <span className="text-xs text-faint pb-2">
          {page === 1 ? "Live: new reports appear at the top." : "Older page: not live."} {liveCount} NLP results received this session · last 24 h
        </span>
      </div>
      {q.isError && <ErrorNote error={q.error} onRetry={() => q.refetch()} />}
      <Panel bodyClass="p-0 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-line">
            <tr>
              <th className="th">Received</th>
              <th className="th">Source</th>
              <th className="th">Report</th>
              <th className="th">Type · confidence</th>
              <th className="th">Severity</th>
              <th className="th">Camera match</th>
            </tr>
          </thead>
          <tbody>
            {q.data
              ? q.data.items.map((rec) => (
                  <ReportRow key={rec.report.report_id} rec={rec} fresh={fresh.has(rec.report.report_id)} cameraName={rec.nlp?.camera_id ? names[rec.nlp.camera_id] : undefined} />
                ))
              : [0, 1, 2, 3, 4, 5].map((n) => (
                  <tr key={n}><td className="td" colSpan={6}><Skeleton className="h-6" /></td></tr>
                ))}
          </tbody>
        </table>
        {q.data?.items.length === 0 && <Empty>No reports match.</Empty>}
        {q.data && <Pager page={page} pageSize={PAGE_SIZE} total={q.data.total} onPage={(p) => set("page", String(p))} />}
      </Panel>
    </div>
  );
}
