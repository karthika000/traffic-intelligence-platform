// Friendly aliases over the generated OpenAPI types (schema.d.ts, `npm run gen:types`).
import type { components } from "./schema";

type S = components["schemas"];
/** Fields the server always fills (they have defaults, so the schema marks them optional). */
type Filled<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;

export type CongestionLevel = S["CongestionLevel"];
export type Severity = S["Severity"];
export type IncidentStatus = S["IncidentStatus"];
export type IncidentType = S["IncidentType"];
export type AnomalyHint = S["AnomalyHint"];
export type ReportSource = S["ReportSource"];

export type Incident = Filled<S["Incident"], "incident_id" | "timestamp" | "message_id">;
export type Summary = Filled<S["Summary"], "summary_id" | "timestamp">;
export type Anomaly = Filled<S["Anomaly"], "anomaly_id" | "timestamp" | "hints">;
export type TrafficMetrics = Filled<S["TrafficMetrics"], "message_id" | "timestamp">;
export type NLPResult = Filled<S["NLPResult"], "timestamp" | "message_id">;
export type TextReport = Filled<S["TextReport"], "report_id" | "timestamp">;

export type CameraStatus = Omit<S["CameraStatus"], "latest_metrics"> & {
  latest_metrics?: TrafficMetrics | null;
};
export type MetricsBucket = S["MetricsBucket"];
export type AnomalyRecord = Omit<S["AnomalyRecord"], "anomaly"> & { anomaly: Anomaly };
export type CameraDetail = Omit<S["CameraDetail"], "camera" | "anomalies"> & {
  camera: CameraStatus;
  anomalies: AnomalyRecord[];
};
export type ReportRecord = { report: TextReport; nlp?: NLPResult | null; unlocated: boolean };
export type Evidence = {
  anomalies: AnomalyRecord[];
  reports: ReportRecord[];
  metrics: TrafficMetrics[];
  weights?: Record<string, number | null>;
};
/** One entry of an incident's published history (fusion_service `details.history`). */
export type IncidentVersion = {
  version: number;
  status: IncidentStatus;
  severity: Severity;
  incident_type: IncidentType;
  fused?: number;
  reasons?: string[];
  at?: string;
};
export type IncidentDetail = {
  incident: Incident;
  resolution?: string | null;
  at_confirmation?: Record<string, unknown> | null;
  versions: IncidentVersion[];
  evidence: Evidence;
  summaries: Summary[];
};
export type CongestionShare = S["CongestionShare"];
export type ModelInfo = S["ModelInfo"];
export type ScenarioInfo = S["ScenarioInfo"];
export type ScenarioRun = S["ScenarioRunRecord"];
export type TriggerRequest = S["TriggerRequest"];
export type TriggerResponse = S["TriggerResponse"];
export type DriftAlert = Filled<S["DriftAlert"], "message_id" | "timestamp" | "details">;
export type DriftFeature = S["DriftFeature"];
export type DriftStatus = Filled<S["DriftStatus"], "features" | "drifted_features" | "quality">;
export type DriftHistoryPoint = Filled<S["DriftHistoryPoint"], "drifted_features">;
export type DriftModeInfo = S["DriftModeInfo"];
/** Fields with server-side defaults may be left out of the request. */
export type DriftModeRequest = Pick<S["DriftModeRequest"], "enabled"> & Partial<S["DriftModeRequest"]>;
export type PromotionDecision = S["PromotionDecision"];
export type RetrainingRecommendation = S["RetrainingRecommendation"];

export type Page<T> = { items: T[]; total: number; page: number; page_size: number };

/** WebSocket envelopes from /ws/live. */
export type LiveData = {
  incidents: Incident;
  summaries: Summary;
  anomalies: Anomaly;
  "traffic.metrics": TrafficMetrics;
  "nlp.results": NLPResult;
  "drift.alerts": DriftAlert;
};
export type LiveTopic = keyof LiveData;
export type LiveEnvelope = {
  [K in LiveTopic]: { type: K; camera_ids: string[]; ts: string; data: LiveData[K] };
}[LiveTopic];
export type ControlMessage =
  | { type: "hello"; client_id: string; heartbeat_seconds?: number }
  | { type: "heartbeat" | "pong"; ts: number }
  | { type: "dropped"; count: number }
  | { type: "subscribed" }
  | { type: "error"; error: string };
