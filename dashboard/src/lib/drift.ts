// Drift monitoring display helpers: the status scale, feature names and alert text.
// Status colours reuse the dashboard's good / warning / critical colours and always come with
// a text label (and an icon in the pill), never colour alone.
import type { DriftAlert, DriftFeature } from "../api/types";

export type DriftStatusName = "ok" | "warning" | "drifted" | "no_data" | "unknown";

export const DRIFT_STATUS: Record<DriftStatusName, { label: string; color: string; icon: string; rank: number }> = {
  ok: { label: "OK", color: "#0ca30c", icon: "✓", rank: 0 },
  warning: { label: "Warning", color: "#fab219", icon: "!", rank: 1 },
  drifted: { label: "Drifted", color: "#d03b3b", icon: "▲", rank: 2 },
  no_data: { label: "No data", color: "#8b8a84", icon: "–", rank: -1 },
  unknown: { label: "Not checked", color: "#8b8a84", icon: "?", rank: -1 },
};

export function driftStatusStyle(status: string | null | undefined) {
  return DRIFT_STATUS[(status ?? "unknown") as DriftStatusName] ?? DRIFT_STATUS.unknown;
}

/** Monitored models (ml-monitor): drift sections appear only on these cards. */
export const MONITORED_MODELS = [
  "vehicle-detector",
  "congestion-classifier",
  "traffic-anomaly-detector",
  "incident-text-classifier",
] as const;

/** Models the retrain CLI (ml.lifecycle.retrain) can retrain. */
export const RETRAINABLE_MODELS = ["congestion-classifier", "traffic-anomaly-detector", "incident-text-classifier"] as const;

const METHOD_LABEL: Record<string, string> = {
  wasserstein: "Wasserstein (norm.)",
  jensenshannon: "Jensen-Shannon",
  model_roc_auc: "classifier AUC",
};
export const methodLabel = (m: string) => METHOD_LABEL[m] ?? m;

export const featureLabel = (name: string) => name.replace(/_/g, " ");

/** "+1.23" style score against its threshold, e.g. "2.41 / 0.10". */
export function scoreText(f: Pick<DriftFeature, "score" | "threshold">): string {
  if (f.score === null || f.score === undefined) return "n/a";
  const digits = Math.abs(f.score) >= 10 ? 1 : 2;
  return `${f.score.toFixed(digits)} / ${f.threshold.toFixed(2)}`;
}

/** Drifted features first, then by how far past (or close to) the threshold they are. */
export function sortFeatures(features: DriftFeature[]): DriftFeature[] {
  const ratio = (f: DriftFeature) => (f.score === null || f.score === undefined ? -1 : f.score / (f.threshold || 1));
  return [...features].sort((a, b) => Number(b.drifted) - Number(a.drifted) || ratio(b) - ratio(a));
}

type AlertFeature = { feature?: string; score?: number | null; threshold?: number | null };

/** One line for a notification: "vehicle-detector v1 drifted: frame brightness 2.41, confidence 0.33". */
export function alertSummary(a: DriftAlert): string {
  const feats = ((a.details?.drifted_features as AlertFeature[] | undefined) ?? [])
    .slice(0, 3)
    .map((f) => `${featureLabel(f.feature ?? "?")}${typeof f.score === "number" ? ` ${f.score.toFixed(2)}` : ""}`);
  const more = ((a.details?.drifted_features as unknown[] | undefined)?.length ?? 0) - feats.length;
  return `${a.model_name} v${a.model_version} drifted (${Math.round(a.drift_score * 100)}% of features)${
    feats.length ? `: ${feats.join(", ")}${more > 0 ? ` +${more}` : ""}` : ""
  }`;
}
