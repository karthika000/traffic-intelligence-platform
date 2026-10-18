// Drift monitoring UI: per-model drift section (status, score over time, drifted features,
// report link, retraining recommendation, last promotion decision), the drift mode switch, and
// the header's drift notifications.
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useDismissRecommendation, useDriftMode, useSetDriftMode } from "../api/queries";
import type { DriftHistoryPoint, DriftStatus } from "../api/types";
import { API_BASE_URL, MLFLOW_URL } from "../lib/config";
import { alertSummary, driftStatusStyle, featureLabel, methodLabel, RETRAINABLE_MODELS, scoreText, sortFeatures } from "../lib/drift";
import { ago, clock, dateTime, zoneName } from "../lib/format";
import { SERIES } from "../lib/scales";
import { useTheme } from "../lib/theme";
import { useLive } from "../store/live";
import { ErrorNote } from "./ui";

const tick = { fill: "var(--chart-axis)", fontSize: 11 };
const runUrl = (exp: string | null | undefined, run: string) =>
  exp ? `${MLFLOW_URL}/#/experiments/${encodeURIComponent(exp)}/runs/${encodeURIComponent(run)}` : `${MLFLOW_URL}/#/runs/${encodeURIComponent(run)}`;

export function DriftPill({ status }: { status?: string | null }) {
  const s = driftStatusStyle(status);
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold whitespace-nowrap"
      style={{ borderColor: s.color, color: "rgb(var(--ink))", background: `${s.color}22` }}
      data-testid="drift-pill"
    >
      <span aria-hidden style={{ color: s.color }}>{s.icon}</span>
      {s.label}
    </span>
  );
}

type ChartPoint = { t: number; share: number; status: string; drifted: string[] };

function StatusDot(props: { cx?: number; cy?: number; payload?: ChartPoint }) {
  const { cx, cy, payload } = props;
  if (cx === undefined || cy === undefined || !payload) return null;
  return <circle cx={cx} cy={cy} r={4} fill={driftStatusStyle(payload.status).color} stroke="rgb(var(--panel))" strokeWidth={2} />;
}

/** Drift share (share of features that drifted) per check, against the drifted threshold. */
export function DriftChart({ points, threshold, hours }: { points: DriftHistoryPoint[]; threshold: number; hours: number }) {
  const theme = useTheme();
  const now = Date.now();
  const data: ChartPoint[] = points.map((p) => ({
    t: new Date(p.window_end).getTime(),
    share: p.drift_share,
    status: p.status,
    drifted: p.drifted_features,
  }));
  return (
    <div className="h-28" role="img" aria-label={`Drift share over the last ${hours} hours`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
          <XAxis dataKey="t" type="number" scale="time" domain={[now - hours * 3_600_000, now]} tickFormatter={(t) => clock(t)} tick={tick} stroke="var(--chart-grid)" minTickGap={40} />
          <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={tick} stroke="var(--chart-grid)" width={44} />
          <ReferenceLine y={threshold} stroke="#d03b3b" strokeDasharray="4 3" ifOverflow="extendDomain" label={{ value: "drifted", position: "insideTopRight", fill: "var(--chart-axis)", fontSize: 10 }} />
          <Tooltip
            labelFormatter={(t) => `${clock(Number(t), true)} ${zoneName(Number(t))}`}
            formatter={(v, _name, item) => {
              const p = item.payload as ChartPoint;
              const feats = p.drifted.length ? ` · ${p.drifted.map(featureLabel).join(", ")}` : "";
              return [`${Math.round(Number(v) * 100)}% (${driftStatusStyle(p.status).label})${feats}`, "drift share"];
            }}
            contentStyle={{ fontSize: 12, color: "rgb(var(--ink))", maxWidth: 320, whiteSpace: "normal" }}
            cursor={{ stroke: "rgb(var(--faint))", strokeWidth: 1 }}
          />
          <Line dataKey="share" stroke={SERIES[theme][0]} strokeWidth={2} dot={<StatusDot />} activeDot={{ r: 5 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** The drift part of a model card. */
export function DriftSection({ status, history, hours }: { status?: DriftStatus; history: DriftHistoryPoint[]; hours: number }) {
  const [all, setAll] = useState(false);
  if (!status) return null;
  const features = sortFeatures(status.features);
  const shown = all ? features : features.slice(0, 5);
  const rec = status.retraining;
  const promo = status.last_promotion;
  return (
    <div className="mt-3 pt-3 border-t border-line text-sm" data-testid="drift-section">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[11px] text-faint uppercase tracking-wide">Drift</span>
        <DriftPill status={status.status} />
        {status.drift_share !== null && status.drift_share !== undefined && (
          <span className="text-xs text-muted tabular-nums">
            {Math.round(status.drift_share * 100)}% of features drifted (threshold {Math.round((status.share_threshold ?? 0) * 100)}%)
          </span>
        )}
        {status.checked_at && <span className="ml-auto text-[11px] text-faint" title={dateTime(status.checked_at, true)}>checked {ago(status.checked_at)}</span>}
      </div>
      {status.status === "unknown" && <p className="mt-2 text-xs text-faint">Not checked yet: ml-monitor runs every check on a schedule.</p>}
      {history.length > 0 && status.share_threshold != null && (
        <div className="mt-2">
          <DriftChart points={history} threshold={status.share_threshold} hours={hours} />
        </div>
      )}
      {shown.length > 0 && (
        <table className="mt-2 w-full text-[11px] tabular-nums">
          <thead>
            <tr className="text-faint text-left">
              <th className="font-normal py-0.5">feature</th>
              <th className="font-normal py-0.5">method</th>
              <th className="font-normal py-0.5 text-right">score / threshold</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((f) => (
              <tr key={f.feature} className="border-t border-line" title={f.note ?? undefined}>
                <td className="py-0.5 pr-2">
                  {f.drifted ? <span className="font-semibold">▲ {featureLabel(f.feature)}</span> : <span className="text-muted">{featureLabel(f.feature)}</span>}
                  <span className="ml-1 text-faint">{f.drift_type}</span>
                </td>
                <td className="py-0.5 pr-2 text-muted">{methodLabel(f.method)}</td>
                <td className={`py-0.5 text-right ${f.drifted ? "font-semibold" : "text-muted"}`}>{scoreText(f)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {features.length > 5 && (
        <button className="mt-1 text-[11px] link" onClick={() => setAll((a) => !a)}>
          {all ? "Show fewer" : `Show all ${features.length} features`}
        </button>
      )}
      {status.quality.length > 0 && (
        <ul className="mt-2 text-[11px] text-muted list-disc pl-4">
          {status.quality.map((q) => (
            <li key={`${q.kind}-${q.target}`}>
              <span className="text-faint">data quality:</span> {q.message}
            </li>
          ))}
        </ul>
      )}
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px]">
        {status.report_url && (
          <a className="link" href={`${API_BASE_URL}${status.report_url}`} target="_blank" rel="noreferrer">
            Evidently report ↗
          </a>
        )}
        {status.mlflow_run_id && (
          <a className="link" href={runUrl(status.mlflow_experiment_id, status.mlflow_run_id)} target="_blank" rel="noreferrer">
            Check in MLflow ↗
          </a>
        )}
        {status.window_start && status.window_end && (
          <span className="text-faint">
            window {clock(status.window_start)}–{clock(status.window_end)} · {status.current_rows ?? 0} rows vs {status.reference_rows ?? 0} reference
          </span>
        )}
      </div>
      {rec && isOpen(rec.status) && (
        <div className="mt-2 rounded-md border border-[#fab219]/60 bg-[#fab219]/10 px-2 py-1.5 text-xs" data-testid="retraining">
          <span className="font-semibold">{rec.status === "retraining" ? "Retraining in progress" : "Retraining recommended"}</span>
          <span className="text-muted"> · {rec.alert_count} alert{rec.alert_count === 1 ? "" : "s"} since {clock(rec.first_alert_at)}</span>
          <div className="text-[11px] text-muted mt-0.5">{rec.reason}</div>
          {(RETRAINABLE_MODELS as readonly string[]).includes(status.model_name) ? (
            <div className="text-[11px] text-faint mt-0.5">
              <code>docker compose run --rm ml-trainer python -m ml.lifecycle.retrain {status.model_name}</code>
            </div>
          ) : (
            <div className="text-[11px] text-faint mt-0.5">This model is pretrained, not trained here: retraining it is outside the retrain CLI.</div>
          )}
          <DismissRecommendation modelName={status.model_name} />
        </div>
      )}
      {promo && (
        <div className="mt-2 text-xs" data-testid="promotion">
          <span className="text-faint">Last promotion decision: </span>
          <span className="font-semibold" style={{ color: promo.decision === "promoted" ? "#0ca30c" : "#d03b3b" }}>
            {promo.decision === "promoted" ? "✓ promoted" : "✕ rejected"}
          </span>
          <span className="text-muted">
            {" "}candidate v{promo.candidate_version ?? "?"} vs champion v{promo.champion_version ?? "–"}
            {promo.decided_at ? ` · ${dateTime(promo.decided_at)}` : ""}
          </span>
          {promo.gate_run_id && (
            <a className="link ml-2 text-[11px]" href={runUrl(null, promo.gate_run_id)} target="_blank" rel="noreferrer">
              gate run ↗
            </a>
          )}
          {(promo.reasons ?? []).length > 0 && (
            <ul className="mt-0.5 text-[11px] text-muted list-disc pl-4">
              {(promo.reasons ?? []).map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      {rec && rec.status === "resolved" && rec.resolved_at && (
        <div className="mt-1 text-[11px] text-faint">Last retraining recommendation resolved {ago(rec.resolved_at)}.</div>
      )}
      {rec && rec.status === "dismissed" && rec.resolved_at && (
        <div className="mt-1 text-[11px] text-faint" data-testid="dismissed">
          Last retraining recommendation dismissed {ago(rec.resolved_at)}
          {typeof rec.resolution?.reason === "string" ? `: ${rec.resolution.reason}` : "."}
        </div>
      )}
    </div>
  );
}

const isOpen = (s: string) => s === "recommended" || s === "retraining";

/** Close an open recommendation without retraining; the reason is required and recorded. */
function DismissRecommendation({ modelName }: { modelName: string }) {
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState("");
  const dismiss = useDismissRecommendation();
  if (!open)
    return (
      <button className="mt-1 text-[11px] link" onClick={() => setOpen(true)}>
        Dismiss…
      </button>
    );
  const ok = reason.trim().length >= 3;
  return (
    <form
      className="mt-1.5 flex items-center gap-1.5"
      onSubmit={(e) => {
        e.preventDefault();
        if (ok) dismiss.mutate({ model: modelName, reason: reason.trim() });
      }}
    >
      <input
        className="input !py-0.5 !text-xs flex-1 min-w-0"
        placeholder="Reason (required)"
        aria-label="Reason for dismissing"
        value={reason}
        maxLength={500}
        onChange={(e) => setReason(e.target.value)}
        autoFocus
      />
      <button className="btn !py-0.5 !text-xs" type="submit" disabled={!ok || dismiss.isPending}>
        {dismiss.isPending ? "Dismissing…" : "Dismiss"}
      </button>
      <button className="text-[11px] link" type="button" onClick={() => setOpen(false)}>
        cancel
      </button>
      {dismiss.isError && <ErrorNote error={dismiss.error} compact />}
    </form>
  );
}

const DURATIONS = [
  { label: "until switched off", value: 0 },
  { label: "30 min", value: 30 },
  { label: "1 hour", value: 60 },
  { label: "2 hours", value: 120 },
];

/** Switch the demo drift mode: night or fog frames in video_ingest, slang reports in event_simulator. */
export function DriftModeControl() {
  const q = useDriftMode();
  const set = useSetDriftMode();
  const [preset, setPreset] = useState<"night" | "fog">("night");
  const [minutes, setMinutes] = useState(60);
  const m = q.data;
  const active = !!m?.active;
  return (
    <div className="flex flex-col gap-2 text-sm" data-testid="drift-mode">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="inline-flex items-center gap-2 font-medium">
          <span className="w-2 h-2 rounded-full" style={{ background: active ? "#d03b3b" : "#8b8a84" }} />
          Drift mode {active ? "ON" : "off"}
        </span>
        {active && m && (
          <span className="text-xs text-muted">
            {m.settings.visual ? `${String(m.settings.preset)} frames` : "frames unchanged"} · {m.settings.text ? "slang reports" : "reports unchanged"}
            {m.expires_at ? ` · switches off at ${clock(m.expires_at)}` : ""}
          </span>
        )}
      </div>
      {!active && (
        <div className="flex items-center gap-2 flex-wrap text-xs">
          <label className="inline-flex items-center gap-1">
            frames
            <select className="input !py-0.5" value={preset} onChange={(e) => setPreset(e.target.value as "night" | "fog")}>
              <option value="night">night (dark + noise)</option>
              <option value="fog">fog (haze + blur + noise)</option>
            </select>
          </label>
          <label className="inline-flex items-center gap-1">
            for
            <select className="input !py-0.5" value={minutes} onChange={(e) => setMinutes(Number(e.target.value))}>
              {DURATIONS.map((d) => (
                <option key={d.value} value={d.value}>{d.label}</option>
              ))}
            </select>
          </label>
        </div>
      )}
      <div className="flex items-center gap-2">
        <button
          className={active ? "btn" : "btn-primary"}
          disabled={set.isPending || !q.data}
          onClick={() =>
            set.mutate(active ? { enabled: false } : { enabled: true, preset, duration_minutes: minutes || undefined })
          }
        >
          {set.isPending ? "Saving…" : active ? "Switch drift mode off" : "Switch drift mode on"}
        </button>
        <span className="text-[11px] text-faint">Services pick the change up within a few seconds.</span>
      </div>
      {(q.isError || set.isError) && <ErrorNote error={q.error ?? set.error} compact />}
    </div>
  );
}

/** Header badge while drift mode is on, with a one-click switch-off. */
export function DriftModeBadge() {
  const q = useDriftMode();
  const set = useSetDriftMode();
  if (!q.data?.active) return null;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-[#d03b3b] bg-[#d03b3b]/15 px-2.5 py-1 text-xs" data-testid="drift-mode-badge">
      <span className="font-medium">Drift mode on</span>
      <button className="link" disabled={set.isPending} onClick={() => set.mutate({ enabled: false })}>
        turn off
      </button>
    </span>
  );
}

const SEEN_KEY = "drift-alerts-seen-at";

function readSeen(): string {
  try {
    return localStorage.getItem(SEEN_KEY) ?? "";
  } catch {
    return "";
  }
}

/** Header notifications: drift alerts from the live feed (seeded from REST). */
export function DriftNotifications() {
  const alerts = useLive((s) => s.driftAlerts);
  const [open, setOpen] = useState(false);
  const [seen, setSeen] = useState(readSeen);
  const ref = useRef<HTMLDivElement>(null);
  const unread = alerts.filter((a) => a.timestamp > seen).length;

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const toggle = () => {
    setOpen((o) => !o);
    if (!open && alerts.length) {
      const newest = alerts[0].timestamp;
      setSeen(newest);
      try {
        localStorage.setItem(SEEN_KEY, newest);
      } catch {
        /* private mode: unread counts just reset on reload */
      }
    }
  };

  return (
    <div className="relative" ref={ref}>
      <button
        className="btn !px-2 !py-1 relative"
        onClick={toggle}
        aria-label={`Drift alerts${unread ? `, ${unread} new` : ""}`}
        title="Drift alerts"
        data-testid="drift-bell"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9" /><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" /></svg>
        {unread > 0 && (
          <span className="absolute -top-1 -right-1 min-w-4 h-4 px-1 rounded-full bg-[#d03b3b] text-white text-[10px] font-semibold leading-4 text-center">
            {unread > 9 ? "9+" : unread}
          </span>
        )}
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-96 max-w-[90vw] panel shadow-lg z-[1100]" role="dialog" aria-label="Drift alerts">
          <div className="flex items-center justify-between px-3 py-2 border-b border-line">
            <span className="panel-title">Drift alerts</span>
            <Link to="/models" className="text-xs link" onClick={() => setOpen(false)}>
              Models →
            </Link>
          </div>
          {alerts.length === 0 ? (
            <p className="px-3 py-4 text-sm text-faint">No drift alerts in the last 24 hours.</p>
          ) : (
            <ul className="max-h-96 overflow-y-auto">
              {alerts.map((a) => (
                <li key={a.message_id} className={`px-3 py-2 border-b border-line text-xs ${a.timestamp > seen ? "" : "opacity-80"}`}>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">▲ {a.model_name}</span>
                    <span className="ml-auto text-faint tabular-nums" title={dateTime(a.timestamp, true)}>{ago(a.timestamp)}</span>
                  </div>
                  <div className="text-muted mt-0.5">{alertSummary(a)}</div>
                  {typeof a.details?.report_url === "string" && (
                    <a className="link text-[11px]" href={`${API_BASE_URL}${a.details.report_url}`} target="_blank" rel="noreferrer">
                      Evidently report ↗
                    </a>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
