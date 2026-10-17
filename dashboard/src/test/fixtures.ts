import type { Incident, LiveEnvelope, Summary, TrafficMetrics } from "../api/types";

export function incident(over: Partial<Incident> = {}): Incident {
  return {
    schema_version: 1,
    message_id: `m-${Math.random()}`,
    timestamp: "2026-10-17T09:00:00Z",
    incident_id: "inc-1",
    camera_id: "cam-hyde-park-corner",
    version: 1,
    status: "suspected",
    incident_type: "accident",
    severity: "high",
    fused_confidence: 0.6,
    vision_degraded: false,
    start_time: "2026-10-17T09:00:00Z",
    suspected_at: "2026-10-17T09:00:00Z",
    title: "Accident at Hyde Park Corner",
    ...over,
  };
}

export function summary(over: Partial<Summary> = {}): Summary {
  return {
    schema_version: 1,
    summary_id: `s-${Math.random()}`,
    timestamp: "2026-10-17T09:00:05Z",
    scope: "incident",
    incident_id: "inc-1",
    incident_version: 1,
    period_start: "2026-10-17T09:00:00Z",
    period_end: "2026-10-17T09:00:05Z",
    headline: "High accident at Hyde Park Corner, suspected",
    text: "...",
    model_name: "template",
    model_version: "1",
    ...over,
  };
}

export function metrics(over: Partial<TrafficMetrics> = {}): TrafficMetrics {
  return {
    schema_version: 1,
    message_id: `mm-${Math.random()}`,
    timestamp: "2026-10-17T09:00:05Z",
    window_start: "2026-10-17T09:00:00Z",
    window_end: "2026-10-17T09:00:05Z",
    camera_id: "cam-hyde-park-corner",
    vehicle_count: 7,
    avg_speed_kmh: 22,
    stopped_vehicles: 0,
    pedestrians_in_roadway: 0,
    frames: 25,
    ...over,
  };
}

export const env = <T extends LiveEnvelope["type"]>(type: T, data: Extract<LiveEnvelope, { type: T }>["data"]) =>
  ({ type, camera_ids: [], ts: "2026-10-17T09:00:00Z", data }) as LiveEnvelope;

/** A controllable stand-in for the browser WebSocket. */
export class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  readyState = 0;
  onopen: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  closed: { code?: number; reason?: string } | null = null;
  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }
  static get last() {
    return FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
  }
  open() {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }
  message(data: unknown) {
    this.onmessage?.({ data: typeof data === "string" ? data : JSON.stringify(data) } as MessageEvent);
  }
  serverClose(code = 1006, reason = "") {
    this.readyState = 3;
    this.onclose?.({ code, reason } as CloseEvent);
  }
  close(code?: number, reason?: string) {
    this.readyState = 3;
    this.closed = { code, reason };
  }
  send() {}
}
