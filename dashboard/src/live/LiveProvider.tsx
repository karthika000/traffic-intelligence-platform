// Starts the one WebSocket connection, seeds the live store from REST, and keeps the two in
// step: after a reconnect it refetches what it may have missed.
import { useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useEffect } from "react";
import { fetchIncident, fetchIncidents } from "../api/queries";
import type { CameraStatus, Incident, Page } from "../api/types";
import { api } from "../api/client";
import { WS_URL } from "../lib/config";
import { LiveSocket } from "../lib/liveSocket";
import { demoStore, evidenceFromRun } from "../store/demo";
import { liveStore } from "../store/live";

async function seed() {
  // Open incidents drive the KPIs and map; the recent page fills the feed. Failures are fine:
  // the views show the API as down and live updates still flow.
  const results = await Promise.allSettled([
    fetchIncidents({ status: "suspected", page_size: 200 }),
    fetchIncidents({ status: "confirmed", page_size: 200 }),
    fetchIncidents({ page_size: 100 }),
  ]);
  const incidents: Incident[] = results.flatMap((r) => (r.status === "fulfilled" ? r.value.items : []));
  liveStore.getState().seedIncidents(incidents);
  try {
    const cams = await api<Page<CameraStatus>>("/api/v1/cameras", { query: { page_size: 200 } });
    liveStore.getState().seedMetrics(cams.items.flatMap((c) => (c.latest_metrics ? [c.latest_metrics] : [])));
  } catch {
    /* shown by the camera views */
  }
}

/** Link tracked demo runs to the incident they produced (see store/demo.ts). */
function useDemoMatching() {
  useEffect(() => {
    const checking = new Set<string>();
    const run = () => {
      demoStore.getState().match(Object.values(liveStore.getState().incidents));
      for (const r of demoStore.getState().runs) {
        if (!r.incidentId || r.linked || checking.has(r.scenarioId)) continue;
        if (Date.now() - r.triggeredAt > 60 * 60_000) continue;
        checking.add(r.scenarioId);
        fetchIncident(r.incidentId)
          .then((d) => {
            if (evidenceFromRun(d, r.scenarioId)) demoStore.getState().confirm(r.scenarioId, d.incident.incident_id, true);
          })
          .catch(() => {})
          .finally(() => checking.delete(r.scenarioId));
      }
    };
    run();
    let last = liveStore.getState().ticks.incidents;
    const unsubLive = liveStore.subscribe((s) => {
      if (s.ticks.incidents !== last) {
        last = s.ticks.incidents;
        run();
      }
    });
    const unsubDemo = demoStore.subscribe((s, prev) => {
      if (s.runs.length !== prev.runs.length) run();
    });
    const timer = setInterval(run, 10_000);
    return () => {
      unsubLive();
      unsubDemo();
      clearInterval(timer);
    };
  }, []);
}

export function LiveProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();

  useEffect(() => {
    void seed();
    const s = liveStore.getState();
    const socket = new LiveSocket({
      url: WS_URL,
      onMessages: (batch) => liveStore.getState().applyBatch(batch),
      onStatus: (info) => liveStore.getState().setStatus(info),
      onControl: (msg) => {
        if (msg.type === "dropped") s.noteDropped(msg.count);
        else s.noteMessage();
      },
      onReconnect: () => {
        void seed();
        void qc.invalidateQueries();
      },
    });
    socket.start();
    const wake = () => socket.reconnectNow();
    const onVisible = () => document.visibilityState === "visible" && wake();
    window.addEventListener("online", wake);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("online", wake);
      document.removeEventListener("visibilitychange", onVisible);
      socket.stop();
    };
  }, [qc]);

  useDemoMatching();
  return <>{children}</>;
}
