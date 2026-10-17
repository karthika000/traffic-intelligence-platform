import L from "leaflet";
import { useEffect, useMemo } from "react";
import { MapContainer, Marker, TileLayer, Tooltip, useMap } from "react-leaflet";
import { useNavigate } from "react-router-dom";
import type { CameraStatus } from "../api/types";
import { num } from "../lib/format";
import { CONGESTION, CONGESTION_LEVELS, congestionStyle, inkOn } from "../lib/scales";

function icon(cam: CameraStatus, open: number) {
  const s = congestionStyle(cam.congestion?.level);
  const badge =
    open > 0
      ? `<span style="position:absolute;top:-7px;right:-9px;min-width:18px;height:18px;padding:0 4px;border-radius:9px;background:#ee4b26;color:#fff;font:600 11px/18px system-ui;text-align:center;border:2px solid rgb(var(--panel))">${open}</span>`
      : "";
  const ring = cam.vision_degraded ? "outline:2px dashed #fab219;outline-offset:2px;" : "";
  return L.divIcon({
    className: "",
    iconSize: [26, 26],
    iconAnchor: [13, 13],
    html: `<div style="position:relative;width:26px;height:26px;border-radius:50%;background:${s.color};border:3px solid rgb(var(--panel));box-shadow:0 1px 6px rgb(0 0 0/.45);${ring}"><svg viewBox="0 0 24 24" width="14" height="14" style="position:absolute;inset:3px" fill="none" stroke="${inkOn(s.color)}" stroke-width="2.4"><rect x="3" y="7" width="13" height="10" rx="2"/><path d="M16 10l5-3v10l-5-3"/></svg>${badge}</div>`,
  });
}

function FitBounds({ cameras }: { cameras: CameraStatus[] }) {
  const map = useMap();
  const key = cameras.map((c) => c.id).join(",");
  useEffect(() => {
    if (!cameras.length) return;
    map.fitBounds(L.latLngBounds(cameras.map((c) => [c.latitude, c.longitude])), { padding: [40, 40], maxZoom: 15 });
    // Re-fit only when the set of cameras changes, not on every refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, map]);
  return null;
}

export function CameraMap({ cameras, openByCamera }: { cameras: CameraStatus[]; openByCamera: Record<string, number> }) {
  const navigate = useNavigate();
  const icons = useMemo(
    () => Object.fromEntries(cameras.map((c) => [c.id, icon(c, openByCamera[c.id] ?? c.open_incidents ?? 0)])),
    [cameras, openByCamera],
  );
  return (
    <div className="relative h-full min-h-[280px]">
      <MapContainer center={[51.5, -0.12]} zoom={12} className="absolute inset-0 rounded-b-lg" scrollWheelZoom={false}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          maxZoom={19}
        />
        <FitBounds cameras={cameras} />
        {cameras.map((c) => (
          <Marker
            key={c.id}
            position={[c.latitude, c.longitude]}
            icon={icons[c.id]}
            eventHandlers={{ click: () => navigate(`/cameras/${c.id}`) }}
            keyboard
            title={c.name}
          >
            <Tooltip direction="top" offset={[0, -14]} className="cam-tip">
              <div className="text-xs">
                <div className="font-semibold">{c.name}</div>
                <div>
                  {congestionStyle(c.congestion?.level).label} · {num(c.latest_metrics?.avg_speed_kmh)} km/h ·{" "}
                  {openByCamera[c.id] ?? c.open_incidents} open
                </div>
                {c.vision_degraded && <div style={{ color: "#fab219" }}>Vision degraded</div>}
              </div>
            </Tooltip>
          </Marker>
        ))}
      </MapContainer>
      <div className="absolute bottom-2 left-2 z-[500] panel px-2 py-1.5 text-[11px] flex flex-wrap gap-x-3 gap-y-1 shadow">
        {CONGESTION_LEVELS.map((l) => (
          <span key={l} className="inline-flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: CONGESTION[l].color }} />
            {CONGESTION[l].label}
          </span>
        ))}
        <span className="inline-flex items-center gap-1">
          <span className="w-3 h-3 rounded-full text-[8px] leading-3 text-center text-white font-bold" style={{ background: "#ee4b26" }}>n</span>
          open incidents
        </span>
      </div>
    </div>
  );
}
