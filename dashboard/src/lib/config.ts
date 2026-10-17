const trim = (url: string) => url.replace(/\/+$/, "");

export const API_BASE_URL = trim(import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8001");

/** ws(s)://<api>/ws/live */
export const WS_URL = `${API_BASE_URL.replace(/^http/, "ws")}/ws/live`;

/**
 * MJPEG streams hold an HTTP/1.1 connection open for as long as they play, and browsers allow
 * only 6 per host. Four camera streams on the API's own host would leave REST calls queueing,
 * so by default streams go to the same API under its other loopback name (localhost <->
 * 127.0.0.1), which the browser counts as a separate host.
 */
export const STREAM_BASE_URL = trim(
  import.meta.env.VITE_STREAM_BASE_URL ??
    (API_BASE_URL.includes("//localhost")
      ? API_BASE_URL.replace("//localhost", "//127.0.0.1")
      : API_BASE_URL.includes("//127.0.0.1")
        ? API_BASE_URL.replace("//127.0.0.1", "//localhost")
        : API_BASE_URL),
);

/** IANA zone every clock time is shown in (DISPLAY_TIMEZONE, the same zone the summarizer
 * writes times in). An unknown zone falls back to Europe/London. */
export const DISPLAY_TIMEZONE = (() => {
  const zone = import.meta.env.VITE_DISPLAY_TIMEZONE?.trim() || "Europe/London";
  try {
    new Intl.DateTimeFormat("en-GB", { timeZone: zone });
    return zone;
  } catch {
    console.warn(`Unknown DISPLAY_TIMEZONE ${zone}; using Europe/London`);
    return "Europe/London";
  }
})();

export const MLFLOW_URL = trim(import.meta.env.VITE_MLFLOW_URL ?? "http://localhost:5000");

export const streamUrl = (cameraId: string, fps = 5) =>
  `${STREAM_BASE_URL}/api/v1/frames/${encodeURIComponent(cameraId)}/stream.mjpg?fps=${fps}`;
