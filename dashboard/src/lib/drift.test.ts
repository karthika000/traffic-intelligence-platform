import { describe, expect, it } from "vitest";
import type { DriftAlert, DriftFeature } from "../api/types";
import { alertSummary, driftStatusStyle, scoreText, sortFeatures } from "./drift";

const feature = (name: string, score: number | null, threshold: number, drifted: boolean): DriftFeature => ({
  feature: name,
  kind: "num",
  drift_type: "data",
  method: "wasserstein",
  score,
  threshold,
  drifted,
});

describe("drift helpers", () => {
  it("maps statuses to labelled styles, unknown for anything else", () => {
    expect(driftStatusStyle("drifted").label).toBe("Drifted");
    expect(driftStatusStyle("ok").label).toBe("OK");
    expect(driftStatusStyle(undefined).label).toBe("Not checked");
    expect(driftStatusStyle("bogus").label).toBe("Not checked");
  });

  it("sorts drifted features first, then by score over threshold", () => {
    const sorted = sortFeatures([
      feature("a", 0.05, 0.1, false),
      feature("b", 0.3, 0.1, true),
      feature("c", null, 0.1, false),
      feature("d", 0.9, 0.1, true),
      feature("e", 0.09, 0.1, false),
    ]);
    expect(sorted.map((f) => f.feature)).toEqual(["d", "b", "e", "a", "c"]);
  });

  it("formats scores against thresholds", () => {
    expect(scoreText({ score: 2.4129, threshold: 0.1 })).toBe("2.41 / 0.10");
    expect(scoreText({ score: null, threshold: 0.1 })).toBe("n/a");
  });

  it("summarises an alert with its top drifted features", () => {
    const alert = {
      message_id: "m",
      timestamp: "2026-10-17T15:00:00Z",
      schema_version: 1,
      window_start: "2026-10-17T14:45:00Z",
      window_end: "2026-10-17T15:00:00Z",
      model_name: "vehicle-detector",
      model_version: "1",
      drift_type: "data",
      metric: "drift_share",
      drift_score: 0.667,
      threshold: 0.3,
      details: {
        drifted_features: [
          { feature: "frame_brightness", score: 2.41 },
          { feature: "confidence", score: 0.33 },
          { feature: "objects_per_frame", score: 0.5 },
          { feature: "no_detections", score: 0.2 },
        ],
      },
    } as DriftAlert;
    expect(alertSummary(alert)).toBe(
      "vehicle-detector v1 drifted (67% of features): frame brightness 2.41, confidence 0.33, objects per frame 0.50 +1",
    );
  });
});
