// Live state fed by the WebSocket (and seeded from REST). Every collection is bounded, and a
// whole batch of messages is applied in one `set`, so subscribers re-render once per batch.
import { useStore } from "zustand";
import { createStore } from "zustand/vanilla";
import type { Anomaly, DriftAlert, Incident, LiveEnvelope, NLPResult, Summary, TrafficMetrics } from "../api/types";
import { capIncidents, type IncidentMap, upsertIncidents, upsertSummaries } from "../lib/incidents";
import type { StatusInfo } from "../lib/liveSocket";

export const LIMITS = { incidents: 300, anomalies: 100, nlp: 200, summaries: 300, driftAlerts: 50 } as const;

export type LiveState = {
  connection: StatusInfo & { lastMessageAt?: number; connectedAt?: number; dropped: number };
  incidents: IncidentMap;
  summaries: Record<string, Summary>;
  metrics: Record<string, TrafficMetrics>;
  anomalies: Anomaly[];
  nlp: NLPResult[];
  /** Newest first, one per alert (message id). */
  driftAlerts: DriftAlert[];
  /** Bumped whenever a batch carried that topic: lets REST views refetch without diffing. */
  ticks: { incidents: number; summaries: number; anomalies: number; nlp: number; drift: number };
};

export type LiveActions = {
  applyBatch: (batch: LiveEnvelope[]) => void;
  seedIncidents: (list: Incident[]) => void;
  seedSummaries: (list: Summary[]) => void;
  seedMetrics: (list: TrafficMetrics[]) => void;
  seedDriftAlerts: (list: DriftAlert[]) => void;
  setStatus: (info: StatusInfo) => void;
  noteMessage: () => void;
  noteDropped: (count: number) => void;
  reset: () => void;
};

export type LiveStore = LiveState & LiveActions;

const initial = (): LiveState => ({
  connection: { status: "connecting", attempt: 0, dropped: 0 },
  incidents: {},
  summaries: {},
  metrics: {},
  anomalies: [],
  nlp: [],
  driftAlerts: [],
  ticks: { incidents: 0, summaries: 0, anomalies: 0, nlp: 0, drift: 0 },
});

function mergeDriftAlerts(list: DriftAlert[], incoming: DriftAlert[]): DriftAlert[] {
  const byId = new Map(list.map((a) => [a.message_id, a]));
  for (const a of incoming) byId.set(a.message_id, a);
  return [...byId.values()].sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1)).slice(0, LIMITS.driftAlerts);
}

function newerMetrics(map: Record<string, TrafficMetrics>, list: TrafficMetrics[]) {
  let out = map;
  for (const m of list) {
    const prev = out[m.camera_id];
    if (prev && prev.window_end >= m.window_end) continue;
    if (out === map) out = { ...map };
    out[m.camera_id] = m;
  }
  return out;
}

function capSummaries(map: Record<string, Summary>): Record<string, Summary> {
  const entries = Object.entries(map);
  if (entries.length <= LIMITS.summaries) return map;
  entries.sort((a, b) => (a[1].timestamp < b[1].timestamp ? 1 : -1));
  return Object.fromEntries(entries.slice(0, LIMITS.summaries));
}

/** Newest first, one entry per anomaly id (start/update/end events replace each other). */
function mergeAnomalies(list: Anomaly[], incoming: Anomaly[]): Anomaly[] {
  const byId = new Map(list.map((a) => [a.anomaly_id, a]));
  for (const a of incoming) byId.set(a.anomaly_id, a);
  return [...byId.values()].sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1)).slice(0, LIMITS.anomalies);
}

export function createLiveStore() {
  return createStore<LiveStore>()((set) => ({
    ...initial(),

    applyBatch: (batch) =>
      set((s) => {
        const by = { incidents: [] as Incident[], summaries: [] as Summary[], anomalies: [] as Anomaly[], metrics: [] as TrafficMetrics[], nlp: [] as NLPResult[], drift: [] as DriftAlert[] };
        for (const env of batch) {
          if (env.type === "incidents") by.incidents.push(env.data);
          else if (env.type === "summaries") by.summaries.push(env.data);
          else if (env.type === "anomalies") by.anomalies.push(env.data);
          else if (env.type === "traffic.metrics") by.metrics.push(env.data);
          else if (env.type === "nlp.results") by.nlp.push(env.data);
          else if (env.type === "drift.alerts") by.drift.push(env.data);
        }
        const next: Partial<LiveState> = {};
        if (by.incidents.length) {
          const merged = upsertIncidents(s.incidents, by.incidents);
          if (merged !== s.incidents) next.incidents = capIncidents(merged, LIMITS.incidents);
        }
        if (by.summaries.length) {
          const merged = upsertSummaries(s.summaries, by.summaries);
          if (merged !== s.summaries) next.summaries = capSummaries(merged);
        }
        if (by.anomalies.length) next.anomalies = mergeAnomalies(s.anomalies, by.anomalies);
        if (by.metrics.length) {
          const merged = newerMetrics(s.metrics, by.metrics);
          if (merged !== s.metrics) next.metrics = merged;
        }
        if (by.nlp.length) next.nlp = [...by.nlp.reverse(), ...s.nlp].slice(0, LIMITS.nlp);
        if (by.drift.length) next.driftAlerts = mergeDriftAlerts(s.driftAlerts, by.drift);
        if (by.incidents.length || by.summaries.length || by.anomalies.length || by.nlp.length || by.drift.length) {
          next.ticks = {
            incidents: s.ticks.incidents + (by.incidents.length ? 1 : 0),
            summaries: s.ticks.summaries + (by.summaries.length ? 1 : 0),
            anomalies: s.ticks.anomalies + (by.anomalies.length ? 1 : 0),
            nlp: s.ticks.nlp + (by.nlp.length ? 1 : 0),
            drift: s.ticks.drift + (by.drift.length ? 1 : 0),
          };
        }
        next.connection = { ...s.connection, lastMessageAt: Date.now() };
        return next;
      }),

    seedIncidents: (list) =>
      set((s) => {
        const merged = upsertIncidents(s.incidents, list);
        return merged === s.incidents ? s : { incidents: capIncidents(merged, LIMITS.incidents) };
      }),

    seedSummaries: (list) =>
      set((s) => {
        const merged = upsertSummaries(s.summaries, list);
        return merged === s.summaries ? s : { summaries: capSummaries(merged) };
      }),

    seedDriftAlerts: (list) => set((s) => ({ driftAlerts: mergeDriftAlerts(s.driftAlerts, list) })),

    seedMetrics: (list) =>
      set((s) => {
        const merged = newerMetrics(s.metrics, list);
        return merged === s.metrics ? s : { metrics: merged };
      }),

    setStatus: (info) =>
      set((s) => ({
        connection: {
          ...s.connection,
          ...info,
          retryAt: info.retryAt,
          reason: info.reason,
          connectedAt: info.status === "open" ? Date.now() : s.connection.connectedAt,
        },
      })),

    noteMessage: () => set((s) => ({ connection: { ...s.connection, lastMessageAt: Date.now() } })),
    noteDropped: (count) => set((s) => ({ connection: { ...s.connection, dropped: s.connection.dropped + count } })),
    reset: () => set(initial()),
  }));
}

export const liveStore = createLiveStore();

export function useLive<T>(selector: (s: LiveStore) => T): T {
  return useStore(liveStore, selector);
}
