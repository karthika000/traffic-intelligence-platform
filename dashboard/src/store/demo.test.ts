import { describe, expect, it } from "vitest";
import { incident } from "../test/fixtures";
import { createDemoStore, matchIncident, mergeRuns, type TrackedRun } from "./demo";

const T0 = Date.parse("2026-10-17T09:00:00Z");
const run: TrackedRun = { scenarioId: "s1", scenario: "multi_vehicle_accident", cameraId: "cam-hyde-park-corner", timeScale: 6, triggeredAt: T0 };

describe("matching a demo run to its incident", () => {
  it("picks the earliest incident on the camera after the trigger", () => {
    const list = [
      incident({ incident_id: "before", suspected_at: "2026-10-17T08:50:00Z", status: "resolved" }),
      incident({ incident_id: "other-cam", camera_id: "cam-old-street", suspected_at: "2026-10-17T09:00:20Z" }),
      incident({ incident_id: "later", suspected_at: "2026-10-17T09:02:00Z" }),
      incident({ incident_id: "first", suspected_at: "2026-10-17T09:00:30Z" }),
    ];
    expect(matchIncident(run, list)?.incident_id).toBe("first");
  });

  it("falls back to an open incident updated after the trigger", () => {
    const list = [incident({ incident_id: "open", suspected_at: "2026-10-17T08:40:00Z", timestamp: "2026-10-17T09:01:00Z" })];
    expect(matchIncident(run, list)?.incident_id).toBe("open");
    expect(matchIncident(run, [incident({ suspected_at: "2026-10-17T08:40:00Z", timestamp: "2026-10-17T08:59:00Z" })])).toBeUndefined();
  });

  it("matches in the store, then keeps a confirmed link", () => {
    const store = createDemoStore([]);
    store.getState().track(run);
    store.getState().match([incident({ incident_id: "a", suspected_at: "2026-10-17T09:00:30Z" })]);
    expect(store.getState().runs[0]).toMatchObject({ incidentId: "a", linked: false });
    store.getState().confirm("s1", "a", true);
    store.getState().match([incident({ incident_id: "b", suspected_at: "2026-10-17T09:00:10Z" })]);
    expect(store.getState().runs[0]).toMatchObject({ incidentId: "a", linked: true });
  });

  it("merges runs from another tab without losing a link", () => {
    const probable = { ...run, incidentId: "a", linked: false };
    const linked = { ...run, incidentId: "a", linked: true };
    const other: TrackedRun = { ...run, scenarioId: "s2", triggeredAt: T0 + 60_000 };
    expect(mergeRuns([linked], [{ ...run }])).toEqual([linked]);
    expect(mergeRuns([probable], [linked])).toEqual([linked]);
    expect(mergeRuns([run], [other]).map((r) => r.scenarioId)).toEqual(["s2", "s1"]); // newest first
  });

  it("picks up what another tab saved", () => {
    const store = createDemoStore([{ ...run }]);
    const theirs = [{ ...run, incidentId: "a", linked: true }];
    window.dispatchEvent(new StorageEvent("storage", { key: "tip.demo.runs", newValue: JSON.stringify(theirs) }));
    expect(store.getState().runs[0]).toMatchObject({ incidentId: "a", linked: true });
  });
});
