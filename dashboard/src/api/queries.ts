import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type {
  CameraDetail,
  CameraStatus,
  CongestionShare,
  DriftAlert,
  DriftHistoryPoint,
  DriftModeInfo,
  DriftModeRequest,
  DriftStatus,
  RetrainingRecommendation,
  Incident,
  IncidentDetail,
  ModelInfo,
  Page,
  ReportRecord,
  ScenarioInfo,
  ScenarioRun,
  TriggerRequest,
  TriggerResponse,
} from "./types";

export type IncidentFilters = {
  status?: string;
  camera_id?: string;
  incident_type?: string;
  severity?: string;
  since?: string;
  until?: string;
  page?: number;
  page_size?: number;
};

export type ReportFilters = {
  camera_id?: string;
  incident_type?: string;
  unlocated?: boolean;
  page?: number;
  page_size?: number;
};

export const keys = {
  health: ["health"] as const,
  cameras: ["cameras"] as const,
  camera: (id: string, minutes: number, resolution: number) => ["camera", id, minutes, resolution] as const,
  congestion: (id: string, minutes: number, bucket: number) => ["congestion", id, minutes, bucket] as const,
  incidents: (f: IncidentFilters) => ["incidents", f] as const,
  incident: (id: string) => ["incident", id] as const,
  reports: (f: ReportFilters) => ["reports", f] as const,
  models: ["models"] as const,
  scenarios: ["scenarios"] as const,
  runs: ["scenario-runs"] as const,
  driftStatus: ["drift-status"] as const,
  driftHistory: (hours: number) => ["drift-history", hours] as const,
  driftAlerts: ["drift-alerts"] as const,
  driftMode: ["drift-mode"] as const,
};

export const useHealth = () =>
  useQuery({
    queryKey: keys.health,
    queryFn: ({ signal }) => api<Record<string, unknown>>("/health", { signal, timeoutMs: 4000 }),
    refetchInterval: 10_000,
    retry: false,
  });

export const useCameras = () =>
  useQuery({
    queryKey: keys.cameras,
    queryFn: ({ signal }) => api<Page<CameraStatus>>("/api/v1/cameras", { signal, query: { page_size: 200 } }),
    select: (p) => p.items,
    refetchInterval: 10_000,
  });

export const useCameraDetail = (id: string, minutes: number, resolution: number, refetchMs: number) =>
  useQuery({
    queryKey: keys.camera(id, minutes, resolution),
    queryFn: ({ signal }) =>
      api<CameraDetail>(`/api/v1/cameras/${encodeURIComponent(id)}`, {
        signal,
        query: { minutes, resolution, anomalies: 50 },
      }),
    refetchInterval: refetchMs,
    placeholderData: keepPreviousData,
  });

export const useCongestionHistory = (id: string, minutes: number, bucket: number, refetchMs: number) =>
  useQuery({
    queryKey: keys.congestion(id, minutes, bucket),
    queryFn: ({ signal }) =>
      api<CongestionShare[]>("/api/v1/analytics/congestion", {
        signal,
        query: { camera_id: id, minutes, bucket_minutes: bucket },
      }),
    refetchInterval: refetchMs,
    placeholderData: keepPreviousData,
  });

export const fetchIncidents = (f: IncidentFilters, signal?: AbortSignal) =>
  api<Page<Incident>>("/api/v1/incidents", { signal, query: f });

export const useIncidents = (f: IncidentFilters, opts: { refetchMs?: number; enabled?: boolean } = {}) =>
  useQuery({
    queryKey: keys.incidents(f),
    queryFn: ({ signal }) => fetchIncidents(f, signal),
    refetchInterval: opts.refetchMs ?? 30_000,
    enabled: opts.enabled ?? true,
    placeholderData: keepPreviousData,
  });

export const fetchIncident = (id: string, signal?: AbortSignal) =>
  api<IncidentDetail>(`/api/v1/incidents/${encodeURIComponent(id)}`, { signal });

export const useIncident = (id: string) =>
  useQuery({
    queryKey: keys.incident(id),
    queryFn: ({ signal }) => fetchIncident(id, signal),
    refetchInterval: 30_000,
  });

export const useReports = (f: ReportFilters) =>
  useQuery({
    queryKey: keys.reports(f),
    queryFn: ({ signal }) => api<Page<ReportRecord>>("/api/v1/reports", { signal, query: f }),
    refetchInterval: 15_000,
    placeholderData: keepPreviousData,
  });

export const useModels = () =>
  useQuery({
    queryKey: keys.models,
    queryFn: ({ signal }) => api<ModelInfo[]>("/api/v1/models", { signal, timeoutMs: 15_000 }),
    refetchInterval: 60_000,
  });

export const useDriftStatus = () =>
  useQuery({
    queryKey: keys.driftStatus,
    queryFn: ({ signal }) => api<DriftStatus[]>("/api/v1/drift/status", { signal }),
    refetchInterval: 30_000,
  });

export const useDriftHistory = (hours: number) =>
  useQuery({
    queryKey: keys.driftHistory(hours),
    queryFn: ({ signal }) => api<DriftHistoryPoint[]>("/api/v1/drift/history", { signal, query: { hours } }),
    refetchInterval: 60_000,
    placeholderData: keepPreviousData,
  });

export const fetchDriftAlerts = (signal?: AbortSignal) =>
  api<DriftAlert[]>("/api/v1/drift/alerts", { signal, query: { hours: 24, limit: 50 } });

export const useDriftMode = () =>
  useQuery({
    queryKey: keys.driftMode,
    queryFn: ({ signal }) => api<DriftModeInfo>("/api/v1/drift/mode", { signal }),
    refetchInterval: 15_000,
  });

export function useSetDriftMode() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: DriftModeRequest) => api<DriftModeInfo>("/api/v1/drift/mode", { method: "POST", body }),
    onSuccess: (info) => qc.setQueryData(keys.driftMode, info),
  });
}

export function useDismissRecommendation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ model, reason }: { model: string; reason: string }) =>
      api<RetrainingRecommendation>(`/api/v1/drift/recommendations/${encodeURIComponent(model)}/dismiss`, {
        method: "POST",
        body: { reason, dismissed_by: "dashboard" },
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.driftStatus }),
  });
}

export const useScenarios = () =>
  useQuery({
    queryKey: keys.scenarios,
    queryFn: ({ signal }) => api<ScenarioInfo[]>("/api/v1/scenarios", { signal }),
    staleTime: 5 * 60_000,
  });

export const useScenarioRuns = (refetchMs: number) =>
  useQuery({
    queryKey: keys.runs,
    queryFn: ({ signal }) => api<Page<ScenarioRun>>("/api/v1/scenarios/runs", { signal, query: { page_size: 20 } }),
    select: (p) => p.items,
    refetchInterval: refetchMs,
  });

export const triggerScenario = (scenario: string, body: TriggerRequest) =>
  api<TriggerResponse>(`/api/v1/scenarios/${encodeURIComponent(scenario)}/trigger`, { method: "POST", body });

export function useTriggerScenario() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ scenario, ...body }: TriggerRequest & { scenario: string }) => triggerScenario(scenario, body),
    onSuccess: () => {
      // The simulator writes the run row a moment after the command; poll it in shortly.
      setTimeout(() => qc.invalidateQueries({ queryKey: keys.runs }), 1500);
    },
  });
}
