import { Link } from "react-router-dom";
import { useShallow } from "zustand/react/shallow";
import { headline } from "../lib/incidents";
import { ago } from "../lib/format";
import { useDemo } from "../store/demo";
import { useLive } from "../store/live";
import { SeverityChip, StatusChip } from "./ui";

/** Overview banner for demo runs triggered in the last 30 minutes, and what they produced. */
export function DemoStrip() {
  const runs = useDemo(useShallow((s) => s.runs.filter((r) => Date.now() - r.triggeredAt < 30 * 60_000)));
  const incidents = useLive((s) => s.incidents);
  const summaries = useLive((s) => s.summaries);
  if (!runs.length) return null;
  return (
    <div className="flex flex-col gap-2">
      {runs.slice(0, 2).map((r) => {
        const inc = r.incidentId ? incidents[r.incidentId] : undefined;
        return (
          <div key={r.scenarioId} className="panel border-accent/50 px-3 py-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm" data-testid="demo-strip">
            <span className="font-semibold text-accent">◆ Demo</span>
            <span>
              <code className="text-xs">{r.scenario}</code> on {r.cameraId} at ×{r.timeScale}, {ago(new Date(r.triggeredAt).toISOString())}
            </span>
            <span className="text-faint">→</span>
            {inc ? (
              <Link to={`/incidents/${inc.incident_id}`} className="flex flex-wrap items-center gap-2 hover:underline">
                <SeverityChip severity={inc.severity} />
                <StatusChip status={inc.status} />
                <span className="font-medium">{headline(inc, summaries[inc.incident_id])}</span>
                <span className="text-xs text-faint">
                  v{inc.version}
                  {r.linked ? " · linked by its reports" : " · probable match"}
                </span>
              </Link>
            ) : (
              <span className="text-muted">waiting for the incident it produces…</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
