import { type ReactNode, useEffect, useRef, useState } from "react";
import { streamUrl } from "../lib/config";

/** True while the page is visible: hidden tabs close their MJPEG connections. */
function usePageVisible() {
  const [visible, setVisible] = useState(() => typeof document === "undefined" || document.visibilityState !== "hidden");
  useEffect(() => {
    const on = () => setVisible(document.visibilityState !== "hidden");
    document.addEventListener("visibilitychange", on);
    return () => document.removeEventListener("visibilitychange", on);
  }, []);
  return visible;
}

/**
 * The annotated MJPEG stream for a camera. The browser decodes it natively in an <img>, so it
 * costs React nothing per frame. On error it retries with backoff (the API returns 404 until
 * cv_inference has written a frame); on unmount it clears `src`, which closes the connection.
 */
export function CameraStream({
  cameraId,
  fps = 5,
  children,
  className = "",
}: {
  cameraId: string;
  fps?: number;
  children?: ReactNode;
  className?: string;
}) {
  const visible = usePageVisible();
  const img = useRef<HTMLImageElement>(null);
  const [failures, setFailures] = useState(0);
  const [nonce, setNonce] = useState(0);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (failures === 0) return;
    const t = setTimeout(() => setNonce((n) => n + 1), Math.min(30_000, 1000 * 2 ** failures));
    return () => clearTimeout(t);
  }, [failures]);

  const src = visible ? `${streamUrl(cameraId, fps)}&r=${nonce}` : "";

  useEffect(() => {
    const el = img.current;
    return () => {
      if (el) el.src = ""; // detach explicitly: some browsers keep a removed <img> streaming
    };
  }, [src]);

  return (
    <div className={`relative overflow-hidden rounded-md bg-black aspect-video ${className}`}>
      {src && (
        <img
          ref={img}
          key={src}
          src={src}
          alt={`Live annotated stream, ${cameraId}`}
          className="absolute inset-0 w-full h-full object-contain"
          onLoad={() => {
            setLoaded(true);
            setFailures(0);
          }}
          onError={() => {
            setLoaded(false);
            setFailures((f) => f + 1);
          }}
        />
      )}
      {!loaded && (
        <div className="absolute inset-0 grid place-items-center text-xs text-white/60">
          {failures > 0 ? "No video yet, retrying…" : visible ? "Connecting to stream…" : "Paused while the tab is hidden"}
        </div>
      )}
      {children}
    </div>
  );
}
