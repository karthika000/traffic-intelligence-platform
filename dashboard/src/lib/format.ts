import { DISPLAY_TIMEZONE } from "./config";

export const titleCase = (s: string | null | undefined) =>
  (s ?? "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

export const num = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(digits);

export const pct = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined ? "–" : `${(v * 100).toFixed(digits)}%`;

// Clock times are shown in DISPLAY_TIMEZONE (24-hour, like the incident summaries), not the
// viewer's own zone, so "since 18:48 BST" in a summary matches the times around it.
const formatters = new Map<string, Intl.DateTimeFormat>();
function fmt(opts: Intl.DateTimeFormatOptions, zone = DISPLAY_TIMEZONE): Intl.DateTimeFormat {
  const key = `${zone}|${JSON.stringify(opts)}`;
  let f = formatters.get(key);
  if (!f) {
    f = new Intl.DateTimeFormat("en-GB", { timeZone: zone, hourCycle: "h23", ...opts });
    formatters.set(key, f);
  }
  return f;
}

const toDate = (v: string | number | Date) => (v instanceof Date ? v : new Date(v));

/** "18:48" (or "18:48:05") in the display zone. */
export function clock(v: string | number | Date, withSeconds = false): string {
  return fmt({ hour: "2-digit", minute: "2-digit", second: withSeconds ? "2-digit" : undefined }).format(toDate(v));
}

/** The display zone's abbreviation at that moment: "BST", "GMT", or an offset like "GMT-4". */
export function zoneName(v: string | number | Date = Date.now(), zone = DISPLAY_TIMEZONE): string {
  return fmt({ timeZoneName: "short" }, zone).formatToParts(toDate(v)).find((p) => p.type === "timeZoneName")?.value ?? zone;
}

const dayKey = (d: Date) => fmt({ year: "numeric", month: "2-digit", day: "2-digit" }).format(d);

export function time(iso: string | null | undefined, withSeconds = false): string {
  return iso ? clock(iso, withSeconds) : "–";
}

/** Time of day for today (in the display zone), date and time otherwise. */
export function dateTime(iso: string | null | undefined, withZone = false): string {
  if (!iso) return "–";
  const d = new Date(iso);
  const text =
    dayKey(d) === dayKey(new Date())
      ? clock(d, true)
      : fmt({ month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(d);
  return withZone ? `${text} ${zoneName(d)}` : text;
}

/** Minutes the zone is ahead of UTC at that instant (60 for BST). */
export function zoneOffsetMinutes(at: Date, zone = DISPLAY_TIMEZONE): number {
  const p = Object.fromEntries(
    fmt({ year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }, zone)
      .formatToParts(at)
      .map((x) => [x.type, x.value]),
  );
  const asUtc = Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour % 24, +p.minute, +p.second);
  return Math.round((asUtc - Math.floor(at.getTime() / 1000) * 1000) / 60_000);
}

/** Midnight today in the display zone, as an instant. */
export function startOfDay(now = new Date(), zone = DISPLAY_TIMEZONE): Date {
  const p = Object.fromEntries(
    fmt({ year: "numeric", month: "2-digit", day: "2-digit" }, zone)
      .formatToParts(now)
      .map((x) => [x.type, x.value]),
  );
  const midnightAsUtc = Date.UTC(+p.year, +p.month - 1, +p.day);
  // The offset at midnight itself, which differs from now's on a clock-change day.
  const guess = new Date(midnightAsUtc - zoneOffsetMinutes(now, zone) * 60_000);
  return new Date(midnightAsUtc - zoneOffsetMinutes(guess, zone) * 60_000);
}

export function ago(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "–";
  const s = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export const shortId = (id: string | null | undefined) => (id ? id.slice(0, 8) : "–");
