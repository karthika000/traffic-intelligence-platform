import { beforeEach, describe, expect, it } from "vitest";
import type { DriftAlert, NLPResult } from "../api/types";
import { env, incident, metrics, summary } from "../test/fixtures";
import { createLiveStore, LIMITS } from "./live";

const nlpResult = (n: number): NLPResult => ({
  schema_version: 1,
  message_id: `x${n}`,
  timestamp: "2026-10-17T09:00:00Z",
  report_id: `r${n}`,
  incident_type: "none",
  incident_confidence: 1,
  stage: "active",
  model_name: "m",
  model_version: "1",
});

const driftAlert = (n: number, model = "vehicle-detector"): DriftAlert => ({
  schema_version: 1,
  message_id: `d${n}`,
  timestamp: `2026-10-17T09:${String(n).padStart(2, "0")}:00Z`,
  window_start: "2026-10-17T08:45:00Z",
  window_end: "2026-10-17T09:00:00Z",
  model_name: model,
  model_version: "1",
  drift_type: "data",
  metric: "drift_share",
  drift_score: 0.67,
  threshold: 0.5,
  details: { drifted_features: [{ feature: "frame_brightness", score: 2.4 }] },
});

describe("live store", () => {
  let store: ReturnType<typeof createLiveStore>;
  beforeEach(() => {
    store = createLiveStore();
  });

  it("applies a batch of mixed topics in one update", () => {
    let updates = 0;
    store.subscribe(() => updates++);
    store.getState().applyBatch([
      env("incidents", incident()),
      env("summaries", summary()),
      env("traffic.metrics", metrics({ vehicle_count: 9 })),
    ]);
    expect(updates).toBe(1);
    const s = store.getState();
    expect(s.incidents["inc-1"].version).toBe(1);
    expect(s.summaries["inc-1"].headline).toContain("Hyde Park");
    expect(s.metrics["cam-hyde-park-corner"].vehicle_count).toBe(9);
    expect(s.ticks).toMatchObject({ incidents: 1, summaries: 1, nlp: 0 });
  });

  it("keeps the newest version of an incident and ignores late older ones", () => {
    const { applyBatch } = store.getState();
    applyBatch([env("incidents", incident({ version: 2, status: "confirmed" }))]);
    applyBatch([env("incidents", incident({ version: 1, status: "suspected" }))]);
    expect(store.getState().incidents["inc-1"]).toMatchObject({ version: 2, status: "confirmed" });
    applyBatch([env("incidents", incident({ version: 3, status: "resolved" }))]);
    expect(store.getState().incidents["inc-1"].status).toBe("resolved");
  });

  it("keeps the latest metrics window per camera", () => {
    const { applyBatch } = store.getState();
    applyBatch([env("traffic.metrics", metrics({ window_end: "2026-10-17T09:00:10Z", vehicle_count: 3 }))]);
    applyBatch([env("traffic.metrics", metrics({ window_end: "2026-10-17T09:00:05Z", vehicle_count: 99 }))]);
    expect(store.getState().metrics["cam-hyde-park-corner"].vehicle_count).toBe(3);
  });

  it("caps list lengths, newest first", () => {
    store.getState().applyBatch(Array.from({ length: LIMITS.nlp + 50 }, (_, n) => env("nlp.results", nlpResult(n))));
    expect(store.getState().nlp).toHaveLength(LIMITS.nlp);
    expect(store.getState().nlp[0].report_id).toBe(`r${LIMITS.nlp + 49}`);
  });

  it("caps incidents but keeps every open one", () => {
    const many = Array.from({ length: LIMITS.incidents + 40 }, (_, n) =>
      env(
        "incidents",
        incident({
          incident_id: `i${n}`,
          status: n < 5 ? "suspected" : "resolved",
          suspected_at: new Date(Date.UTC(2026, 9, 17, 0, n)).toISOString(),
        }),
      ),
    );
    store.getState().applyBatch(many);
    const ids = Object.keys(store.getState().incidents);
    expect(ids).toHaveLength(LIMITS.incidents);
    for (let n = 0; n < 5; n++) expect(ids).toContain(`i${n}`); // the oldest, but open
  });

  it("does not notify subscribers when a seed changes nothing", () => {
    store.getState().seedIncidents([incident({ version: 2 })]);
    let updates = 0;
    store.subscribe(() => updates++);
    store.getState().seedIncidents([incident({ version: 1 })]);
    expect(updates).toBe(0);
  });

  it("tracks connection status and dropped counts", () => {
    store.getState().setStatus({ status: "reconnecting", attempt: 2, retryAt: 123, reason: "closed (1006)" });
    expect(store.getState().connection).toMatchObject({ status: "reconnecting", attempt: 2, retryAt: 123 });
    store.getState().setStatus({ status: "open", attempt: 0 });
    expect(store.getState().connection.status).toBe("open");
    expect(store.getState().connection.retryAt).toBeUndefined();
    expect(store.getState().connection.connectedAt).toBeTypeOf("number");
    store.getState().noteDropped(4);
    store.getState().noteDropped(1);
    expect(store.getState().connection.dropped).toBe(5);
  });

  it("keeps drift alerts newest first, once each, capped", () => {
    store.getState().seedDriftAlerts([driftAlert(1)]);
    store.getState().applyBatch([env("drift.alerts", driftAlert(2, "incident-text-classifier")), env("drift.alerts", driftAlert(1))]);
    const s = store.getState();
    expect(s.driftAlerts.map((a) => a.message_id)).toEqual(["d2", "d1"]);
    expect(s.ticks.drift).toBe(1);
    store.getState().seedDriftAlerts(Array.from({ length: 60 }, (_, i) => driftAlert(i + 3)));
    expect(store.getState().driftAlerts).toHaveLength(LIMITS.driftAlerts);
  });
});
