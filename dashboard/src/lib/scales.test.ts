import { describe, expect, it } from "vitest";
import {
  CONGESTION,
  CONGESTION_LEVELS,
  congestionStyle,
  dominantLevel,
  inkOn,
  SEVERITIES,
  SEVERITY,
  severityStyle,
  UNKNOWN,
} from "./scales";

describe("congestion colour scale", () => {
  it("maps every level to its own colour and label, in order", () => {
    expect(congestionStyle("free_flow")).toMatchObject({ label: "Free flow", rank: 0 });
    expect(congestionStyle("moderate")).toMatchObject({ label: "Moderate", rank: 1 });
    expect(congestionStyle("heavy")).toMatchObject({ label: "Heavy", rank: 2 });
    expect(congestionStyle("standstill")).toMatchObject({ label: "Standstill", rank: 3 });
    expect(new Set(CONGESTION_LEVELS.map((l) => CONGESTION[l].color)).size).toBe(4);
  });

  it("falls back to the neutral unknown style", () => {
    expect(congestionStyle(null)).toBe(UNKNOWN);
    expect(congestionStyle(undefined)).toBe(UNKNOWN);
    expect(congestionStyle("gridlock")).toBe(UNKNOWN);
  });

  it("picks the dominant level of a bucket", () => {
    expect(dominantLevel({ free_flow: 0.2, moderate: 0.1, heavy: 0.7, standstill: 0 })).toBe("heavy");
    expect(dominantLevel({})).toBeNull();
  });
});

describe("severity colour scale", () => {
  it("maps every severity to its own colour and label, in order", () => {
    expect(severityStyle("low")).toMatchObject({ label: "Low", rank: 0 });
    expect(severityStyle("medium")).toMatchObject({ label: "Medium", rank: 1 });
    expect(severityStyle("high")).toMatchObject({ label: "High", rank: 2 });
    expect(severityStyle("critical")).toMatchObject({ label: "Critical", rank: 3 });
    expect(new Set(SEVERITIES.map((s) => SEVERITY[s].color)).size).toBe(4);
    expect(severityStyle("bogus")).toBe(UNKNOWN);
  });

  it("never reuses a congestion colour, so the scales are not confused side by side", () => {
    const congestion = new Set(Object.values(CONGESTION).map((c) => c.color));
    for (const s of Object.values(SEVERITY)) expect(congestion.has(s.color)).toBe(false);
  });
});

describe("inkOn", () => {
  it("chooses dark text on light fills and white on dark fills", () => {
    expect(inkOn("#fab219")).toBe("#111110");
    expect(inkOn("#d03b3b")).toBe("#ffffff");
  });
});
