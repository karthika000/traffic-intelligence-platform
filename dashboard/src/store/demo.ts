// Scenario runs triggered from this browser, and the incident each one produced.
//
// Incidents don't carry a scenario id, but the reports fusion attaches do (TextReport.scenario_id).
// So a run is first matched to a *candidate*: the earliest incident on its camera that started
// after the trigger. The candidate becomes *linked* once its detail shows an evidence report
// from that run; the incident detail view and the demo panel both check.
import { useStore } from "zustand";
import { createStore } from "zustand/vanilla";
import type { Incident, IncidentDetail } from "../api/types";
import { startedAt } from "../lib/incidents";

export type TrackedRun = {
  scenarioId: string;
  scenario: string;
  cameraId: string;
  timeScale: number;
  triggeredAt: number; // ms epoch
  incidentId?: string;
  /** true once an evidence report of the incident carries this run's scenario_id */
  linked?: boolean;
};

const KEY = "tip.demo.runs";
const MAX = 10;
/** Anomalies can lead the trigger slightly (clock skew, a window already in flight). */
export const MATCH_SLACK_MS = 15_000;

function load(): TrackedRun[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as TrackedRun[]).slice(0, MAX) : [];
  } catch {
    return [];
  }
}

function save(runs: TrackedRun[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(runs));
  } catch {
    /* storage unavailable: tracking still works for this page view */
  }
}

/**
 * Combine two views of the tracked runs (this tab's and another tab's). A run keeps what either
 * side learned: a confirmed link wins over a probable match, which wins over nothing.
 */
export function mergeRuns(mine: TrackedRun[], theirs: TrackedRun[]): TrackedRun[] {
  const rank = (r: TrackedRun) => (r.linked ? 2 : r.incidentId ? 1 : 0);
  const byId = new Map<string, TrackedRun>();
  for (const r of [...mine, ...theirs]) {
    const prev = byId.get(r.scenarioId);
    byId.set(r.scenarioId, !prev || rank(r) > rank(prev) ? r : prev);
  }
  return [...byId.values()].sort((a, b) => b.triggeredAt - a.triggeredAt).slice(0, MAX);
}

/**
 * The earliest incident on the run's camera that started at or after the trigger. Failing that,
 * an incident already open there that was updated after the trigger: fusion attaches the run's
 * reports to it instead of opening a second one.
 */
export function matchIncident(run: TrackedRun, incidents: Incident[]): Incident | undefined {
  const from = run.triggeredAt - MATCH_SLACK_MS;
  const t = (iso: string) => new Date(iso).getTime();
  const onCamera = incidents.filter((i) => i.camera_id === run.cameraId);
  const fresh = onCamera
    .filter((i) => t(startedAt(i)) >= from)
    .sort((a, b) => t(startedAt(a)) - t(startedAt(b)))[0];
  if (fresh) return fresh;
  return onCamera
    .filter((i) => i.status !== "resolved" && t(i.timestamp) >= run.triggeredAt)
    .sort((a, b) => t(b.timestamp) - t(a.timestamp))[0];
}

export const evidenceFromRun = (detail: IncidentDetail, scenarioId: string) =>
  detail.evidence.reports.some((r) => r.report.scenario_id === scenarioId);

type DemoStore = {
  runs: TrackedRun[];
  track: (run: TrackedRun) => void;
  /** Re-match every unlinked run against the incidents we know of. */
  match: (incidents: Incident[]) => void;
  confirm: (scenarioId: string, incidentId: string, linked: boolean) => void;
  clear: () => void;
};

export function createDemoStore(initial: TrackedRun[] = load()) {
  const store = createStore<DemoStore>()((set) => ({
    runs: initial,
    track: (run) => set((s) => ({ runs: [run, ...s.runs.filter((r) => r.scenarioId !== run.scenarioId)].slice(0, MAX) })),
    match: (incidents) =>
      set((s) => {
        let changed = false;
        const runs = s.runs.map((r) => {
          if (r.linked) return r;
          const m = matchIncident(r, incidents);
          if (!m || m.incident_id === r.incidentId) return r;
          changed = true;
          return { ...r, incidentId: m.incident_id, linked: false };
        });
        return changed ? { runs } : s;
      }),
    confirm: (scenarioId, incidentId, linked) =>
      set((s) => ({
        runs: s.runs.map((r) =>
          r.scenarioId === scenarioId ? { ...r, incidentId: linked ? incidentId : r.incidentId, linked } : r,
        ),
      })),
    clear: () => set({ runs: [] }),
  }));
  store.subscribe((s) => save(s.runs));
  // Other tabs of the dashboard track and link runs too: merge what they save.
  if (typeof window !== "undefined") {
    window.addEventListener("storage", (e) => {
      if (e.key !== KEY) return;
      try {
        const theirs = e.newValue ? (JSON.parse(e.newValue) as TrackedRun[]) : [];
        const merged = e.newValue === null ? [] : mergeRuns(store.getState().runs, theirs);
        if (JSON.stringify(merged) !== JSON.stringify(store.getState().runs)) store.setState({ runs: merged });
      } catch {
        /* unreadable value from another tab: keep ours */
      }
    });
  }
  return store;
}

export const demoStore = createDemoStore();
export const useDemo = <T,>(selector: (s: DemoStore) => T): T => useStore(demoStore, selector);

/** incident id -> the tracked run that produced it */
export function demoIncidentIds(runs: TrackedRun[]): Map<string, TrackedRun> {
  return new Map(runs.filter((r) => r.incidentId).map((r) => [r.incidentId as string, r]));
}
