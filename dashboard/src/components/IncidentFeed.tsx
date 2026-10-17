import { memo, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { Incident, Summary } from "../api/types";
import { ago, titleCase } from "../lib/format";
import { headline, startedAt } from "../lib/incidents";
import { severityStyle } from "../lib/scales";
import type { TrackedRun } from "../store/demo";
import { Empty, SeverityChip, StatusChip, Tag } from "./ui";

type RowProps = {
  incident: Incident;
  summary?: Summary;
  cameraName?: string;
  demo?: TrackedRun;
  now: number;
};

/** One feed row. Memoised: it only re-renders when its own incident, summary or label changes. */
export const IncidentRow = memo(function IncidentRow({ incident: i, summary, cameraName, demo, now }: RowProps) {
  // Flash when a newer version arrives in place (not on first render).
  const seen = useRef(i.version);
  const [flash, setFlash] = useState(0);
  useEffect(() => {
    if (i.version !== seen.current) {
      seen.current = i.version;
      setFlash((n) => n + 1);
    }
  }, [i.version]);

  return (
    <li
      data-testid="incident-row"
      data-incident={i.incident_id}
      data-version={i.version}
      className={`relative border-b border-line last:border-0 ${demo ? "ring-2 ring-inset ring-accent bg-accent/5" : ""}`}
    >
      <span className="absolute left-0 inset-y-0 w-1" style={{ background: severityStyle(i.severity).color }} aria-hidden />
      <Link
        to={`/incidents/${i.incident_id}`}
        key={flash}
        className={`block pl-4 pr-3 py-2.5 hover:bg-raised ${flash ? "animate-flash" : ""}`}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="font-medium text-sm leading-snug" data-testid="incident-headline">
            {headline(i, summary)}
          </div>
          <span className="text-[11px] text-faint whitespace-nowrap tabular-nums" title={startedAt(i)}>
            {ago(startedAt(i), now)}
          </span>
        </div>
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          <SeverityChip severity={i.severity} />
          <StatusChip status={i.status} />
          <Tag>{titleCase(i.incident_type)}</Tag>
          {cameraName && <span className="text-[11px] text-muted truncate">{cameraName}</span>}
          <span className="text-[11px] text-faint ml-auto tabular-nums" title="incident version">
            v{i.version}
          </span>
          {i.vision_degraded && <Tag title="Vision evidence was degraded">vision degraded</Tag>}
          {demo && (
            <span className="text-[11px] font-semibold text-accent">
              ◆ demo: {demo.scenario}
              {demo.linked ? "" : " (probable)"}
            </span>
          )}
        </div>
      </Link>
    </li>
  );
});

export function IncidentFeed({
  incidents,
  summaries,
  cameraNames,
  demoByIncident,
  emptyText = "No incidents yet. New ones appear here as fusion publishes them.",
}: {
  incidents: Incident[];
  summaries: Record<string, Summary>;
  cameraNames: Record<string, string>;
  demoByIncident?: Map<string, TrackedRun>;
  emptyText?: string;
}) {
  // One clock for every row's "x ago", ticking every 15 s.
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 15_000);
    return () => clearInterval(t);
  }, []);
  if (!incidents.length) return <Empty>{emptyText}</Empty>;
  return (
    <ul aria-label="Incident feed">
      {incidents.map((i) => (
        <IncidentRow
          key={i.incident_id}
          incident={i}
          summary={summaries[i.incident_id]}
          cameraName={cameraNames[i.camera_id] ?? i.camera_id}
          demo={demoByIncident?.get(i.incident_id)}
          now={now}
        />
      ))}
    </ul>
  );
}
