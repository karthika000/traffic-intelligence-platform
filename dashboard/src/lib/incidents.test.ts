import { describe, expect, it } from "vitest";
import { incident, summary } from "../test/fixtures";
import { capIncidents, feed, headline, isNewer, upsertIncidents, upsertSummaries } from "./incidents";

describe("incident feed logic", () => {
  it("updates an incident in place when a newer version arrives", () => {
    let map = upsertIncidents({}, [incident({ version: 1 })]);
    map = upsertIncidents(map, [incident({ version: 2, status: "confirmed", severity: "critical" })]);
    expect(Object.keys(map)).toEqual(["inc-1"]);
    expect(map["inc-1"]).toMatchObject({ version: 2, status: "confirmed", severity: "critical" });
  });

  it("ignores stale or duplicate versions and returns the same object", () => {
    const map = upsertIncidents({}, [incident({ version: 3 })]);
    expect(upsertIncidents(map, [incident({ version: 2 })])).toBe(map);
    expect(upsertIncidents(map, [incident({ version: 3 })])).toBe(map);
  });

  it("breaks version ties by timestamp", () => {
    const a = incident({ version: 2, timestamp: "2026-10-17T09:00:00Z" });
    expect(isNewer(incident({ version: 2, timestamp: "2026-10-17T09:00:01Z" }), a)).toBe(true);
    expect(isNewer(incident({ version: 2, timestamp: "2026-10-17T08:59:59Z" }), a)).toBe(false);
  });

  it("orders the feed newest first and limits its length", () => {
    const map = upsertIncidents({}, [
      incident({ incident_id: "old", suspected_at: "2026-10-17T08:00:00Z" }),
      incident({ incident_id: "new", suspected_at: "2026-10-17T10:00:00Z" }),
      incident({ incident_id: "mid", suspected_at: "2026-10-17T09:00:00Z" }),
    ]);
    expect(feed(map, 10).map((i) => i.incident_id)).toEqual(["new", "mid", "old"]);
    expect(feed(map, 2).map((i) => i.incident_id)).toEqual(["new", "mid"]);
  });

  it("keeps an updated incident in its place in the order", () => {
    let map = upsertIncidents({}, [
      incident({ incident_id: "a", suspected_at: "2026-10-17T10:00:00Z" }),
      incident({ incident_id: "b", suspected_at: "2026-10-17T09:00:00Z" }),
    ]);
    map = upsertIncidents(map, [incident({ incident_id: "b", version: 5, suspected_at: "2026-10-17T09:00:00Z", status: "confirmed" })]);
    expect(feed(map, 10).map((i) => `${i.incident_id}:${i.version}`)).toEqual(["a:1", "b:5"]);
  });

  it("caps the map but keeps every open incident", () => {
    const resolved = Array.from({ length: 10 }, (_, n) =>
      incident({ incident_id: `r${n}`, status: "resolved", suspected_at: `2026-10-17T0${n}:00:00Z` }),
    );
    const map = upsertIncidents({}, [...resolved, incident({ incident_id: "open", suspected_at: "2026-10-16T00:00:00Z" })]);
    expect(Object.keys(capIncidents(map, 4)).sort()).toEqual(["open", "r7", "r8", "r9"]);
  });

  it("uses the latest summary's headline, falling back to the title", () => {
    let s = upsertSummaries({}, [summary({ incident_version: 1, headline: "v1 headline" })]);
    s = upsertSummaries(s, [summary({ incident_version: 2, headline: "v2 headline" })]);
    s = upsertSummaries(s, [summary({ incident_version: 1, headline: "late v1", timestamp: "2026-10-17T10:00:00Z" })]);
    expect(headline(incident(), s["inc-1"])).toBe("v2 headline");
    expect(headline(incident(), undefined)).toBe("Accident at Hyde Park Corner");
    expect(upsertSummaries({}, [summary({ scope: "camera" })])).toEqual({});
  });
});
