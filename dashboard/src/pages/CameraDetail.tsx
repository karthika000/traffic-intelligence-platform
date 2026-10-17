import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useCameraDetail, useCongestionHistory, useIncidents } from "../api/queries";
import type { Anomaly } from "../api/types";
import { CameraStream } from "../components/CameraStream";
import { CongestionTimeline, type Point, TimeChart } from "../components/charts";
import { IncidentFeed } from "../components/IncidentFeed";
import { CongestionChip, Empty, ErrorNote, Panel, Segmented, Skeleton, Tag } from "../components/ui";
import { dateTime, num, titleCase } from "../lib/format";
import { feed } from "../lib/incidents";
import { liveStore, useLive } from "../store/live";

type RangeKey = "15m" | "1h" | "6h" | "24h";
// resolution: metrics bucket in seconds; cBucket: congestion bucket in minutes.
const RANGES: Record<RangeKey, { minutes: number; resolution: number; refetch: number; cBucket: number }> = {
  "15m": { minutes: 15, resolution: 15, refetch: 10_000, cBucket: 1 },
  "1h": { minutes: 60, resolution: 60, refetch: 20_000, cBucket: 5 },
  "6h": { minutes: 360, resolution: 300, refetch: 60_000, cBucket: 15 },
  "24h": { minutes: 1440, resolution: 900, refetch: 120_000, cBucket: 60 },
};

function AnomalyRow({ a, status }: { a: Anomaly; status?: string }) {
  const over = a.threshold ? a.score / a.threshold : null;
  return (
    <li className="px-3 py-2 border-b border-line last:border-0 text-sm">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-medium">{titleCase(a.anomaly_type)}</span>
        {a.hints.map((h) => (
          <Tag key={h}>{titleCase(h)}</Tag>
        ))}
        {a.scope === "system" && <Tag>system-wide</Tag>}
        <span className="ml-auto text-[11px] text-faint tabular-nums">{dateTime(a.started_at ?? a.window_end)}</span>
      </div>
      <div className="mt-1 flex flex-wrap gap-x-4 text-xs text-muted tabular-nums">
        {a.scope === "system" ? (
          // A system episode's score is how many cameras dropped, its threshold the minimum.
          <span title="cameras with a frame or confidence drop in the same check, and the minimum for a system-wide episode">
            <span className="text-ink font-medium">{num(a.score)}</span> cameras degraded (minimum {num(a.threshold)}) · lowest{" "}
            {a.metric_name} ratio {num(a.observed_value, 2)}
          </span>
        ) : (
          <>
            <span title="anomaly score vs the model's threshold">
              score <span className="text-ink font-medium">{num(a.score, 2)}</span> / {num(a.threshold, 2)}
              {over !== null && <span className={over >= 1 ? "text-[#ec835a]" : ""}> ({num(over, 1)}×)</span>}
            </span>
            <span>
              {a.metric_name}: {num(a.observed_value, 1)}
              {a.expected_value !== null && a.expected_value !== undefined && ` vs ${num(a.expected_value, 1)} expected`}
            </span>
          </>
        )}
        <span>{status ?? a.event}</span>
        {a.congestion_level && <CongestionChip level={a.congestion_level} />}
      </div>
    </li>
  );
}

export function CameraDetail() {
  const { id = "" } = useParams();
  const [range, setRange] = useState<RangeKey>("1h");
  const r = RANGES[range];
  const detail = useCameraDetail(id, r.minutes, r.resolution, r.refetch);
  const congestion = useCongestionHistory(id, r.minutes, r.cBucket, Math.max(10_000, r.refetch));
  const incidentsQ = useIncidents({ camera_id: id, page_size: 30 });
  useEffect(() => {
    if (incidentsQ.data) liveStore.getState().seedIncidents(incidentsQ.data.items);
  }, [incidentsQ.data]);

  const live = useLive((s) => s.metrics[id]);
  const liveAnomalies = useLive((s) => s.anomalies);
  const incidents = useLive((s) => s.incidents);
  const summaries = useLive((s) => s.summaries);

  const cam = detail.data?.camera;
  const latest = live ?? cam?.latest_metrics;

  const points: Point[] = useMemo(
    () =>
      (detail.data?.metrics ?? []).map((b) => ({
        t: new Date(b.bucket_start).getTime(),
        vehicles: b.vehicle_count_mean,
        avg: b.avg_speed_kmh,
        p85: b.p85_speed_kmh,
        density: b.density_mean,
        stopped: b.stopped_vehicles_max,
      })),
    [detail.data?.metrics],
  );

  // REST anomalies, with live updates for this camera merged on top (same id replaces).
  const anomalies = useMemo(() => {
    const byId = new Map<string, { a: Anomaly; status?: string }>();
    for (const rec of detail.data?.anomalies ?? []) byId.set(rec.anomaly.anomaly_id, { a: rec.anomaly, status: rec.status });
    for (const a of liveAnomalies)
      if (a.camera_id === id || a.affected_cameras?.includes(id)) {
        const prev = byId.get(a.anomaly_id);
        if (!prev || a.timestamp >= prev.a.timestamp) byId.set(a.anomaly_id, { a, status: a.event === "end" ? "ended" : "active" });
      }
    return [...byId.values()].sort((x, y) => (x.a.timestamp < y.a.timestamp ? 1 : -1)).slice(0, 30);
  }, [detail.data?.anomalies, liveAnomalies, id]);

  const camIncidents = useMemo(
    () => feed(Object.fromEntries(Object.entries(incidents).filter(([, i]) => i.camera_id === id)), 30),
    [incidents, id],
  );

  if (detail.isError && !detail.data) {
    return (
      <div className="flex flex-col gap-3">
        <Link to="/" className="text-sm link">← Overview</Link>
        <ErrorNote error={detail.error} onRetry={() => detail.refetch()} />
      </div>
    );
  }

  const chart = (title: string, body: React.ReactNode) => (
    <Panel title={title} className="h-[220px]" bodyClass="p-2">
      {detail.data ? points.length ? body : <Empty>No metrics in this range.</Empty> : <Skeleton className="h-full" />}
    </Panel>
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <Link to="/" className="text-sm link">← Overview</Link>
        <h1 className="text-lg font-semibold">{cam?.name ?? id}</h1>
        <span className="text-sm text-muted">{cam?.location_name}</span>
        {cam && <CongestionChip level={cam.congestion?.level} solid />}
        {cam?.vision_degraded && <Tag title={cam.vision_degraded_reasons?.join(", ")}>vision degraded</Tag>}
        <div className="ml-auto">
          <Segmented
            label="Time range"
            value={range}
            onChange={setRange}
            options={(Object.keys(RANGES) as RangeKey[]).map((k) => ({ value: k, label: k }))}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">
        <div className="xl:col-span-3">
          <CameraStream cameraId={id} className="panel !rounded-lg">
            <div className="absolute inset-x-0 bottom-0 flex flex-wrap items-end gap-5 p-3 bg-gradient-to-t from-black/80 to-transparent text-white tabular-nums">
              {[
                ["vehicles", latest?.vehicle_count],
                ["km/h avg", num(latest?.avg_speed_kmh)],
                ["km/h p85", num(latest?.p85_speed_kmh)],
                ["density", num(latest?.density, 1)],
                ["stopped", latest?.stopped_vehicles],
                ["pedestrians in road", latest?.pedestrians_in_roadway],
              ].map(([label, v]) => (
                <span key={label as string}>
                  <span className="text-xl font-semibold">{v ?? "–"}</span>
                  <span className="text-[11px] text-white/70 ml-1">{label}</span>
                </span>
              ))}
              <span className="ml-auto text-[11px] text-white/60">window {dateTime(latest?.window_end)}</span>
            </div>
          </CameraStream>
        </div>
        <div className="xl:col-span-2 flex flex-col gap-4 min-w-0">
          <Panel title="Congestion level">
            {congestion.data ? (
              <CongestionTimeline buckets={congestion.data} rangeMinutes={r.minutes} bucketMinutes={r.cBucket} />
            ) : congestion.isError ? (
              <ErrorNote error={congestion.error} compact />
            ) : (
              <Skeleton className="h-7" />
            )}
            <div className="mt-2 text-[11px] text-faint">Dominant level per {r.cBucket === 1 ? "1-minute" : `${r.cBucket}-minute`} bucket from anomaly_service's congestion checks.</div>
          </Panel>
          <Panel title="Incidents at this camera" bodyClass="p-0 overflow-y-auto max-h-[300px]">
            <IncidentFeed incidents={camIncidents} summaries={summaries} cameraNames={{}} emptyText="No incidents at this camera." />
          </Panel>
        </div>
      </div>

      {detail.isError && <ErrorNote error={detail.error} compact onRetry={() => detail.refetch()} />}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {chart("Vehicle count", <TimeChart data={points} series={[{ key: "vehicles", label: "Vehicles" }]} rangeMinutes={r.minutes} />)}
        {chart(
          "Speed (km/h)",
          <TimeChart data={points} series={[{ key: "avg", label: "Average" }, { key: "p85", label: "85th percentile" }]} unit="km/h" rangeMinutes={r.minutes} />,
        )}
        {chart("Density (vehicles in roadway)", <TimeChart data={points} series={[{ key: "density", label: "Density" }]} rangeMinutes={r.minutes} digits={2} />)}
        {chart("Stopped vehicles (max)", <TimeChart data={points} series={[{ key: "stopped", label: "Stopped" }]} rangeMinutes={r.minutes} bars digits={0} />)}
      </div>
      <div className="text-[11px] text-faint -mt-2">
        {detail.data && `${points.length} buckets of ${detail.data.resolution_seconds} s · refreshes every ${r.refetch / 1000} s`}
      </div>

      <Panel title="Recent anomalies" bodyClass="p-0 max-h-[360px] overflow-y-auto">
        {anomalies.length ? (
          <ul>
            {anomalies.map(({ a, status }) => (
              <AnomalyRow key={a.anomaly_id} a={a} status={status} />
            ))}
          </ul>
        ) : (
          <Empty>No anomalies in this range.</Empty>
        )}
      </Panel>
    </div>
  );
}
