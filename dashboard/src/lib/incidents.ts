// Incident feed logic: fusion_service republishes an incident under the same id with a higher
// version on every change, so the feed is a map keyed by id where only newer versions win.
import type { Incident, Summary } from "../api/types";

export type IncidentMap = Record<string, Incident>;

/** Is `next` a later state of the incident than `prev`? */
export function isNewer(next: Incident, prev: Incident | undefined): boolean {
  if (!prev) return true;
  if (next.version !== prev.version) return next.version > prev.version;
  return (next.timestamp ?? "") > (prev.timestamp ?? "");
}

/** Apply incidents (REST seed or live) to the map. Returns the same object when nothing changed. */
export function upsertIncidents(map: IncidentMap, incoming: Incident[]): IncidentMap {
  let out = map;
  for (const inc of incoming) {
    if (!inc.incident_id || !isNewer(inc, out[inc.incident_id])) continue;
    if (out === map) out = { ...map };
    out[inc.incident_id] = inc;
  }
  return out;
}

export const startedAt = (i: Incident) => i.suspected_at ?? i.start_time;
export const isOpen = (i: Incident) => i.status !== "resolved";

/** Newest first by when the incident started; ties by id so the order is stable. */
export function sortIncidents(list: Incident[]): Incident[] {
  return [...list].sort((a, b) => {
    const d = new Date(startedAt(b)).getTime() - new Date(startedAt(a)).getTime();
    return d !== 0 ? d : a.incident_id < b.incident_id ? -1 : 1;
  });
}

/**
 * Keep the map bounded: every open incident (they drive the KPIs and the map), plus the most
 * recent resolved ones up to `cap` entries in total.
 */
export function capIncidents(map: IncidentMap, cap: number): IncidentMap {
  const all = Object.values(map);
  if (all.length <= cap) return map;
  const open = all.filter(isOpen);
  const resolved = sortIncidents(all.filter((i) => !isOpen(i))).slice(0, Math.max(0, cap - open.length));
  return Object.fromEntries([...open, ...resolved].map((i) => [i.incident_id, i]));
}

export function feed(map: IncidentMap, limit: number): Incident[] {
  return sortIncidents(Object.values(map)).slice(0, limit);
}

/** Latest incident summary per incident: higher incident_version wins, then the newer one. */
export function upsertSummaries(map: Record<string, Summary>, incoming: Summary[]): Record<string, Summary> {
  let out = map;
  for (const s of incoming) {
    if (s.scope !== "incident" || !s.incident_id) continue;
    const prev = out[s.incident_id];
    const newer =
      !prev ||
      (s.incident_version ?? 0) > (prev.incident_version ?? 0) ||
      ((s.incident_version ?? 0) === (prev.incident_version ?? 0) && s.timestamp > prev.timestamp);
    if (!newer) continue;
    if (out === map) out = { ...map };
    out[s.incident_id] = s;
  }
  return out;
}

/** Headline to show for an incident: its latest summary's headline, else the fused title. */
export const headline = (i: Incident, summary?: Summary) => summary?.headline || i.title;
