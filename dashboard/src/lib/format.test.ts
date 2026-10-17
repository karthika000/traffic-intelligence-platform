import { describe, expect, it } from "vitest";
import { DISPLAY_TIMEZONE } from "./config";
import { clock, startOfDay, zoneName, zoneOffsetMinutes } from "./format";

describe("display time zone", () => {
  it("defaults to Europe/London", () => {
    expect(DISPLAY_TIMEZONE).toBe("Europe/London");
  });

  it("shows 24-hour London time whatever the viewer's zone", () => {
    expect(clock("2026-10-17T10:02:57Z")).toBe("11:02");
    expect(clock("2026-10-17T10:02:57Z", true)).toBe("11:02:57");
    expect(clock("2026-12-01T17:48:00Z")).toBe("17:48");
  });

  it("names the zone like the summaries do", () => {
    expect(zoneName("2026-10-17T10:00:00Z")).toBe("BST");
    expect(zoneName("2026-12-01T10:00:00Z")).toBe("GMT");
    expect(zoneName("2026-10-17T10:00:00Z", "America/New_York")).toBe("GMT-4");
  });

  it("computes the zone offset", () => {
    expect(zoneOffsetMinutes(new Date("2026-10-17T10:00:00Z"))).toBe(60);
    expect(zoneOffsetMinutes(new Date("2026-12-01T10:00:00Z"))).toBe(0);
    expect(zoneOffsetMinutes(new Date("2026-10-17T10:00:00Z"), "Asia/Kolkata")).toBe(330);
  });

  it("finds midnight in the display zone, including on clock-change days", () => {
    expect(startOfDay(new Date("2026-10-17T10:00:00Z")).toISOString()).toBe("2026-10-16T23:00:00.000Z");
    expect(startOfDay(new Date("2026-10-16T23:30:00Z")).toISOString()).toBe("2026-10-16T23:00:00.000Z");
    expect(startOfDay(new Date("2026-12-01T10:00:00Z")).toISOString()).toBe("2026-12-01T00:00:00.000Z");
    // 25 October 2026: clocks go back at 02:00 BST, after midnight (still BST).
    expect(startOfDay(new Date("2026-10-25T20:00:00Z")).toISOString()).toBe("2026-10-24T23:00:00.000Z");
    // 29 March 2026: clocks go forward at 01:00 GMT; midnight was GMT.
    expect(startOfDay(new Date("2026-03-29T20:00:00Z")).toISOString()).toBe("2026-03-29T00:00:00.000Z");
  });
});
