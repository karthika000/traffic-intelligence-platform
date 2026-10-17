import type { ReactNode } from "react";
import { ApiError } from "../api/client";
import { titleCase } from "../lib/format";
import { congestionStyle, inkOn, severityStyle, STATUS_LABEL } from "../lib/scales";

export function Panel({
  title,
  actions,
  children,
  className = "",
  bodyClass = "p-3",
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section className={`panel flex flex-col min-w-0 ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-2 px-3 py-2 border-b border-line">
          {typeof title === "string" ? <h2 className="panel-title">{title}</h2> : title}
          {actions}
        </header>
      )}
      <div className={`flex-1 min-h-0 ${bodyClass}`}>{children}</div>
    </section>
  );
}

/** Four pips filled up to the rank: a second channel beside colour for ordinal levels. */
function Pips({ rank, color }: { rank: number; color: string }) {
  return (
    <span className="inline-flex gap-[2px]" aria-hidden>
      {[0, 1, 2, 3].map((i) => (
        <span
          key={i}
          className="w-[3px] h-2.5 rounded-sm"
          style={{ background: i <= rank ? color : "rgb(var(--line))" }}
        />
      ))}
    </span>
  );
}

export function CongestionChip({ level, solid = false }: { level?: string | null; solid?: boolean }) {
  const s = congestionStyle(level);
  if (solid)
    return (
      <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[11px] font-semibold" style={{ background: s.color, color: inkOn(s.color) }}>
        {s.label}
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium whitespace-nowrap" title={`Congestion: ${s.label}`}>
      <span className="w-2 h-2 rounded-full" style={{ background: s.color }} />
      {s.label}
    </span>
  );
}

export function SeverityChip({ severity }: { severity?: string | null }) {
  const s = severityStyle(severity);
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 text-[11px] font-semibold whitespace-nowrap"
      style={{ borderColor: `${s.color}80`, background: `${s.color}1f` }}
      title={`Severity: ${s.label}`}
    >
      <Pips rank={s.rank} color={s.color} />
      {s.label}
    </span>
  );
}

const STATUS_CLASS: Record<string, string> = {
  suspected: "border-dashed border-faint text-muted",
  confirmed: "border-ink/60 text-ink bg-ink/5",
  resolved: "border-line text-faint line-through decoration-faint/60",
};

export function StatusChip({ status }: { status: string }) {
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[11px] font-medium whitespace-nowrap ${STATUS_CLASS[status] ?? ""}`}>
      {STATUS_LABEL[status] ?? titleCase(status)}
    </span>
  );
}

export function Tag({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span title={title} className="inline-flex items-center rounded bg-raised border border-line px-1.5 py-0.5 text-[11px] text-muted whitespace-nowrap">
      {children}
    </span>
  );
}

export function Kpi({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: string }) {
  return (
    <div className="panel px-4 py-3 min-w-0">
      <div className="panel-title truncate">{label}</div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="text-3xl font-semibold tabular-nums" style={tone ? { color: tone } : undefined}>
          {value}
        </span>
        {sub && <span className="text-xs text-muted truncate">{sub}</span>}
      </div>
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-raised ${className}`} />;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="py-8 text-center text-sm text-faint">{children}</div>;
}

export function ErrorNote({ error, onRetry, compact }: { error: unknown; onRetry?: () => void; compact?: boolean }) {
  const msg =
    error instanceof ApiError
      ? error.unreachable
        ? `${error.message}. Retrying in the background.`
        : error.message
      : error instanceof Error
        ? error.message
        : "Something went wrong";
  return (
    <div className={`flex items-center justify-between gap-3 rounded-md border border-[#d03b3b]/40 bg-[#d03b3b]/10 text-sm ${compact ? "px-2 py-1" : "px-3 py-2"}`} role="alert">
      <span>{msg}</span>
      {onRetry && (
        <button className="btn !py-0.5 text-xs" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-md border border-line bg-raised p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={`px-2.5 py-1 text-xs font-medium rounded ${o.value === value ? "bg-panel text-ink shadow-sm" : "text-muted hover:text-ink"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-muted min-w-0">
      <span className="font-medium">{label}</span>
      {children}
    </label>
  );
}

export function Pager({ page, pageSize, total, onPage }: { page: number; pageSize: number; total: number; onPage: (p: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="flex items-center justify-between gap-2 px-3 py-2 text-xs text-muted border-t border-line">
      <span>
        {total === 0 ? "No results" : `${(page - 1) * pageSize + 1}–${Math.min(total, page * pageSize)} of ${total}`}
      </span>
      <div className="flex items-center gap-1">
        <button className="btn !py-0.5" disabled={page <= 1} onClick={() => onPage(page - 1)}>
          Previous
        </button>
        <span className="px-2">
          {page} / {pages}
        </span>
        <button className="btn !py-0.5" disabled={page >= pages} onClick={() => onPage(page + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}

/** Horizontal bar for a 0..1 score. */
export function ScoreBar({ value, color = "rgb(var(--accent))" }: { value: number | null | undefined; color?: string }) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="relative h-1.5 w-14 rounded-full bg-line overflow-hidden">
        <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${v * 100}%`, background: color }} />
      </span>
      <span className="tabular-nums text-xs">{value === null || value === undefined ? "–" : value.toFixed(2)}</span>
    </span>
  );
}
