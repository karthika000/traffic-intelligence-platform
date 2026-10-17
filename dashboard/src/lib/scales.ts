// The two state scales used everywhere: on the map, chips, KPI tiles, tables and charts.
// Congestion uses the status ramp (green -> amber -> orange -> red); severity uses a separate
// blue -> violet -> magenta -> red ramp, so the two never read as the same thing side by side.
// Colours are mode-invariant (each clears 3:1 on the dark surface), and every use pairs them
// with a text label, never colour alone.
import type { CongestionLevel, Severity } from "../api/types";

export type ScaleEntry = { label: string; color: string; rank: number };

export const CONGESTION_LEVELS = ["free_flow", "moderate", "heavy", "standstill"] as const satisfies readonly CongestionLevel[];
export const SEVERITIES = ["low", "medium", "high", "critical"] as const satisfies readonly Severity[];

export const CONGESTION: Record<CongestionLevel, ScaleEntry> = {
  free_flow: { label: "Free flow", color: "#0ca30c", rank: 0 },
  moderate: { label: "Moderate", color: "#fab219", rank: 1 },
  heavy: { label: "Heavy", color: "#ec835a", rank: 2 },
  standstill: { label: "Standstill", color: "#d03b3b", rank: 3 },
};

export const SEVERITY: Record<Severity, ScaleEntry> = {
  low: { label: "Low", color: "#5aa4ea", rank: 0 },
  medium: { label: "Medium", color: "#6f5ddb", rank: 1 },
  high: { label: "High", color: "#dc5aaa", rank: 2 },
  critical: { label: "Critical", color: "#ee4b26", rank: 3 },
};

/** Grey for "no data yet" / unknown values. */
export const UNKNOWN: ScaleEntry = { label: "Unknown", color: "#8b8a84", rank: -1 };

export function congestionStyle(level: string | null | undefined): ScaleEntry {
  return (level && CONGESTION[level as CongestionLevel]) || UNKNOWN;
}

export function severityStyle(severity: string | null | undefined): ScaleEntry {
  return (severity && SEVERITY[severity as Severity]) || UNKNOWN;
}

/** Text colour that reads on top of a scale colour used as a fill. */
export function inkOn(hex: string): string {
  const n = parseInt(hex.slice(1), 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return lum > 0.3 ? "#111110" : "#ffffff";
}

export const STATUS_LABEL: Record<string, string> = {
  suspected: "Suspected",
  confirmed: "Confirmed",
  resolved: "Resolved",
};

/** Series colours for charts (categorical slots 1 and 2), per theme. */
export const SERIES = {
  light: ["#2a78d6", "#eb6834"],
  dark: ["#3987e5", "#d95926"],
} as const;

/** The level with the largest share in a bucket. */
export function dominantLevel(shares: Record<string, number> | undefined): string | null {
  let best: string | null = null;
  let max = 0;
  for (const level of CONGESTION_LEVELS) {
    const v = shares?.[level] ?? 0;
    if (v > max) {
      max = v;
      best = level;
    }
  }
  return best;
}
