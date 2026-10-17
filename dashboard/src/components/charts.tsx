import { memo } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { CongestionShare } from "../api/types";
import { CONGESTION, CONGESTION_LEVELS, congestionStyle, dominantLevel, SERIES } from "../lib/scales";
import { useTheme } from "../lib/theme";
import { clock, zoneName } from "../lib/format";

export type Point = { t: number } & Record<string, number | null>;
export type SeriesDef = { key: string; label: string };

const tick = { fill: "var(--chart-axis)", fontSize: 11 };

const fmtTime = (t: number) => clock(t);

/** A small time-series chart: one y-axis, 2px lines, crosshair tooltip, legend only for 2+ series. */
export const TimeChart = memo(function TimeChart({
  data,
  series,
  unit,
  rangeMinutes,
  bars = false,
  digits = 1,
}: {
  data: Point[];
  series: SeriesDef[];
  unit?: string;
  rangeMinutes: number;
  bars?: boolean;
  digits?: number;
}) {
  const theme = useTheme();
  const colors = SERIES[theme];
  const now = Date.now();
  const domain: [number, number] = [now - rangeMinutes * 60_000, now];
  const common = {
    data,
    margin: { top: 8, right: 12, bottom: 0, left: -8 },
  };
  const axes = (
    <>
      <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
      <XAxis dataKey="t" type="number" scale="time" domain={domain} tickFormatter={fmtTime} tick={tick} stroke="var(--chart-grid)" minTickGap={40} />
      <YAxis tick={tick} stroke="var(--chart-grid)" width={44} allowDecimals={digits > 0} />
      <Tooltip
        labelFormatter={(t) => `${clock(Number(t), true)} ${zoneName(Number(t))}`}
        formatter={(v, name) => [v === null || v === undefined ? "–" : `${Number(v).toFixed(digits)}${unit ? ` ${unit}` : ""}`, name]}
        contentStyle={{ fontSize: 12, color: "rgb(var(--ink))" }}
        cursor={bars ? { fill: "rgb(var(--line) / 0.5)" } : { stroke: "rgb(var(--faint))", strokeWidth: 1 }}
      />
      {series.length > 1 && <Legend iconType="plainline" wrapperStyle={{ fontSize: 11, color: "rgb(var(--muted))" }} />}
    </>
  );
  return (
    <ResponsiveContainer width="100%" height="100%">
      {bars ? (
        <BarChart {...common} barCategoryGap={1}>
          {axes}
          {series.map((s, i) => (
            <Bar key={s.key} dataKey={s.key} name={s.label} fill={colors[i]} radius={[2, 2, 0, 0]} isAnimationActive={false} maxBarSize={12} />
          ))}
        </BarChart>
      ) : (
        <LineChart {...common}>
          {axes}
          {series.map((s, i) => (
            <Line key={s.key} dataKey={s.key} name={s.label} stroke={colors[i]} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} activeDot={{ r: 4, strokeWidth: 2, stroke: "rgb(var(--panel))" }} />
          ))}
        </LineChart>
      )}
    </ResponsiveContainer>
  );
});


/** A strip of buckets coloured by their dominant congestion level, with the shares on hover. */
export function CongestionTimeline({ buckets, rangeMinutes, bucketMinutes }: { buckets: CongestionShare[]; rangeMinutes: number; bucketMinutes: number }) {
  const end = Date.now();
  const start = end - rangeMinutes * 60_000;
  const visible = buckets.filter((b) => new Date(b.bucket_start).getTime() + bucketMinutes * 60_000 > start);
  return (
    <div>
      <div className="relative h-7 rounded bg-raised overflow-hidden" role="img" aria-label="Congestion level over time">
        {visible.map((b) => {
          const t0 = Math.max(start, new Date(b.bucket_start).getTime());
          const t1 = Math.min(end, new Date(b.bucket_start).getTime() + bucketMinutes * 60_000);
          const level = dominantLevel(b.shares as Record<string, number>);
          const shares = CONGESTION_LEVELS.map((l) => `${CONGESTION[l].label} ${Math.round(((b.shares as Record<string, number>)[l] ?? 0) * 100)}%`).join(", ");
          return (
            <div
              key={b.bucket_start}
              className="absolute inset-y-0 border-r-2 border-panel"
              style={{ left: `${((t0 - start) / (end - start)) * 100}%`, width: `${((t1 - t0) / (end - start)) * 100}%`, background: congestionStyle(level).color }}
              title={`${clock(b.bucket_start)}: ${shares}`}
            />
          );
        })}
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-faint tabular-nums">
        <span>{clock(start)} {zoneName(start)}</span>
        <span className="flex gap-3">
          {CONGESTION_LEVELS.map((l) => (
            <span key={l} className="inline-flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm" style={{ background: CONGESTION[l].color }} />
              {CONGESTION[l].label}
            </span>
          ))}
        </span>
        <span>now</span>
      </div>
    </div>
  );
}
