import { useEffect, useMemo } from "react";
import { Link } from "react-router-dom";
import { useShallow } from "zustand/react/shallow";
import { useCameras, useIncidents } from "../api/queries";
import type { CameraStatus, TrafficMetrics } from "../api/types";
import { CameraMap } from "../components/CameraMap";
import { CameraStream } from "../components/CameraStream";
import { IncidentFeed } from "../components/IncidentFeed";
import { CongestionChip, ErrorNote, Kpi, Panel, Skeleton } from "../components/ui";
import { DemoStrip } from "../components/DemoStrip";
import { num, startOfDay, zoneName } from "../lib/format";
import { feed, isOpen } from "../lib/incidents";
import { demoIncidentIds, useDemo } from "../store/demo";
import { liveStore, useLive } from "../store/live";

const FEED_LIMIT = 40;

function CameraTile({ cam, metrics }: { cam: CameraStatus; metrics?: TrafficMetrics | null }) {
  return (
    <Link to={`/cameras/${cam.id}`} className="panel block overflow-hidden hover:border-accent/60 group">
      <CameraStream cameraId={cam.id} className="rounded-none">
        <div className="absolute inset-x-0 top-0 flex items-start justify-between gap-2 p-2 bg-gradient-to-b from-black/70 to-transparent text-white">
          <span className="text-xs font-semibold drop-shadow truncate">{cam.name}</span>
          <CongestionChip level={cam.congestion?.level} solid />
        </div>
        <div className="absolute inset-x-0 bottom-0 flex items-end gap-4 p-2 bg-gradient-to-t from-black/75 to-transparent text-white tabular-nums">
          <span>
            <span className="text-lg font-semibold">{metrics?.vehicle_count ?? "–"}</span>
            <span className="text-[11px] text-white/70 ml-1">vehicles</span>
          </span>
          <span>
            <span className="text-lg font-semibold">{num(metrics?.avg_speed_kmh)}</span>
            <span className="text-[11px] text-white/70 ml-1">km/h avg</span>
          </span>
          {!!metrics?.stopped_vehicles && <span className="text-[11px] font-semibold text-[#fab219]">{metrics.stopped_vehicles} stopped</span>}
          {cam.vision_degraded && <span className="ml-auto text-[11px] font-semibold text-[#fab219]" title={cam.vision_degraded_reasons?.join(", ")}>vision degraded</span>}
        </div>
      </CameraStream>
    </Link>
  );
}

export function Overview() {
  const cameras = useCameras();
  const today = useMemo(() => startOfDay().toISOString(), []);
  // Today's incidents (for "confirmed today") go into the live store with everything else.
  const todays = useIncidents({ since: today, page_size: 200 }, { refetchMs: 60_000 });
  useEffect(() => {
    if (todays.data) liveStore.getState().seedIncidents(todays.data.items);
  }, [todays.data]);

  const incidentsMap = useLive((s) => s.incidents);
  const summaries = useLive((s) => s.summaries);
  const metrics = useLive((s) => s.metrics);
  const runs = useDemo(useShallow((s) => s.runs));

  const list = useMemo(() => Object.values(incidentsMap), [incidentsMap]);
  const feedItems = useMemo(() => feed(incidentsMap, FEED_LIMIT), [incidentsMap]);
  const openByCamera = useMemo(() => {
    const out: Record<string, number> = {};
    for (const i of list) if (isOpen(i)) out[i.camera_id] = (out[i.camera_id] ?? 0) + 1;
    return out;
  }, [list]);
  const cams = useMemo(() => cameras.data ?? [], [cameras.data]);
  const names = useMemo(() => Object.fromEntries(cams.map((c) => [c.id, c.name])), [cams]);
  const demoMap = useMemo(() => demoIncidentIds(runs), [runs]);

  const kpis = useMemo(() => {
    const open = list.filter(isOpen).length;
    const confirmedToday = list.filter((i) => i.confirmed_at && i.confirmed_at >= today).length;
    const degraded = cams.filter((c) => c.vision_degraded).length;
    const speeds = cams.map((c) => (metrics[c.id] ?? c.latest_metrics)?.avg_speed_kmh).filter((v): v is number => typeof v === "number");
    const avgSpeed = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null;
    return { open, confirmedToday, degraded, avgSpeed, speedN: speeds.length };
  }, [list, cams, metrics, today]);

  return (
    <div className="flex flex-col gap-4">
      <DemoStrip />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Kpi label="Open incidents" value={kpis.open} tone={kpis.open ? "#ee4b26" : undefined} sub="suspected + confirmed" />
        <Kpi label="Confirmed today" value={kpis.confirmedToday} sub={`since midnight ${zoneName()}`} />
        <Kpi
          label="Cameras degraded"
          value={cameras.data ? kpis.degraded : "–"}
          tone={kpis.degraded ? "#fab219" : undefined}
          sub={cameras.data ? `of ${cams.length}` : undefined}
        />
        <Kpi label="Average speed" value={num(kpis.avgSpeed, 1)} sub={`km/h across ${kpis.speedN} cameras`} />
      </div>

      {cameras.isError && !cameras.data && <ErrorNote error={cameras.error} onRetry={() => cameras.refetch()} />}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Panel title="Camera map" className="lg:col-span-2 h-[420px]" bodyClass="p-0">
          {cameras.data ? <CameraMap cameras={cams} openByCamera={openByCamera} /> : <Skeleton className="h-full rounded-none" />}
        </Panel>
        <Panel
          title="Live incident feed"
          actions={
            <Link to="/incidents" className="text-xs link">
              All incidents
            </Link>
          }
          className="h-[420px]"
          bodyClass="p-0 overflow-y-auto"
        >
          <IncidentFeed incidents={feedItems} summaries={summaries} cameraNames={names} demoByIncident={demoMap} />
        </Panel>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
        {cameras.data
          ? cams.map((c) => <CameraTile key={c.id} cam={c} metrics={metrics[c.id] ?? c.latest_metrics} />)
          : [0, 1, 2, 3].map((i) => <Skeleton key={i} className="aspect-video" />)}
      </div>
    </div>
  );
}
