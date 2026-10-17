import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useHealth } from "../api/queries";
import { API_BASE_URL, DISPLAY_TIMEZONE } from "../lib/config";
import { clock, zoneName } from "../lib/format";
import { themeStore, useTheme } from "../lib/theme";
import { useLive } from "../store/live";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/incidents", label: "Incidents" },
  { to: "/reports", label: "Reports" },
  { to: "/models", label: "Models" },
  { to: "/demo", label: "Demo control" },
];

function Dot({ color, pulse }: { color: string; pulse?: boolean }) {
  return (
    <span className="relative inline-flex w-2 h-2">
      {pulse && <span className="absolute inset-0 rounded-full animate-ping opacity-60" style={{ background: color }} />}
      <span className="relative w-2 h-2 rounded-full" style={{ background: color }} />
    </span>
  );
}

function useNow(ms: number) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(t);
  }, [ms]);
  return now;
}

export function ConnectionStatus() {
  const status = useLive((s) => s.connection.status);
  const attempt = useLive((s) => s.connection.attempt);
  const retryAt = useLive((s) => s.connection.retryAt);
  const reason = useLive((s) => s.connection.reason);
  const dropped = useLive((s) => s.connection.dropped);
  const now = useNow(1000);
  const view =
    status === "open"
      ? { color: "#0ca30c", text: "Live", pulse: true }
      : status === "connecting"
        ? { color: "#fab219", text: "Connecting…" }
        : status === "reconnecting"
          ? {
              color: "#ec835a",
              text: retryAt && retryAt > now ? `Reconnecting in ${Math.ceil((retryAt - now) / 1000)}s` : `Reconnecting (attempt ${attempt})`,
            }
          : { color: "#8b8a84", text: "Offline" };
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-line bg-raised px-2.5 py-1 text-xs"
      title={`Live updates (WebSocket)${reason ? ` — last close: ${reason}` : ""}${dropped ? ` — ${dropped} messages dropped by the server` : ""}`}
      data-testid="ws-status"
    >
      <Dot color={view.color} pulse={view.pulse} />
      <span className="font-medium">{view.text}</span>
    </span>
  );
}

function ApiStatus() {
  const { isError, isSuccess } = useHealth();
  const color = isError ? "#d03b3b" : isSuccess ? "#0ca30c" : "#8b8a84";
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-line bg-raised px-2.5 py-1 text-xs" title={API_BASE_URL}>
      <Dot color={color} />
      <span className="font-medium">API {isError ? "down" : isSuccess ? "up" : "…"}</span>
    </span>
  );
}

/** The display zone's current time: every time on the dashboard is in this zone. */
function ZoneClock() {
  const now = useNow(15_000);
  return (
    <span className="hidden md:inline text-xs text-muted tabular-nums whitespace-nowrap" title={`All times are shown in ${DISPLAY_TIMEZONE}`}>
      {clock(now)} {zoneName(now)}
    </span>
  );
}

function ThemeToggle() {
  const theme = useTheme();
  return (
    <button
      className="btn !px-2 !py-1"
      onClick={() => themeStore.getState().toggle()}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
    >
      {theme === "dark" ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
      )}
    </button>
  );
}

function ApiDownBanner() {
  const { isError } = useHealth();
  if (!isError) return null;
  return (
    <div className="bg-[#d03b3b]/15 border-b border-[#d03b3b]/40 text-sm px-4 py-1.5" role="status">
      The API at {API_BASE_URL} isn't responding. Showing the last data received; views refresh on their own when it's back.
    </div>
  );
}

export function Layout() {
  return (
    <div className="min-h-full flex flex-col">
      <header className="sticky top-0 z-[1000] bg-panel/95 backdrop-blur border-b border-line">
        <div className="flex items-center gap-4 px-4 h-12">
          <div className="flex items-center gap-2 font-semibold whitespace-nowrap">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="rgb(var(--accent))" strokeWidth="2"><rect x="7" y="2" width="10" height="20" rx="3" /><circle cx="12" cy="7" r="1.6" /><circle cx="12" cy="12" r="1.6" /><circle cx="12" cy="17" r="1.6" /></svg>
            <span className="hidden lg:inline">Traffic Intelligence</span>
          </div>
          <nav className="flex items-center gap-1 overflow-x-auto">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                className={({ isActive }) =>
                  `px-2.5 py-1.5 rounded-md text-sm whitespace-nowrap ${isActive ? "bg-raised text-ink font-medium" : "text-muted hover:text-ink"}`
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <ZoneClock />
            <ApiStatus />
            <ConnectionStatus />
            <ThemeToggle />
          </div>
        </div>
        <ApiDownBanner />
      </header>
      <main className="flex-1 p-4 min-w-0">
        <Outlet />
      </main>
    </div>
  );
}
