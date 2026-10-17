/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  /** Host for MJPEG streams; defaults to the API on 127.0.0.1 (see src/lib/config.ts). */
  readonly VITE_STREAM_BASE_URL?: string;
  readonly VITE_MLFLOW_URL?: string;
  /** IANA zone for clock times (docker-compose passes DISPLAY_TIMEZONE). */
  readonly VITE_DISPLAY_TIMEZONE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
