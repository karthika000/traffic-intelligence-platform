import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // File-change events don't cross the Windows -> Docker bind mount, so poll inside containers.
    watch: { usePolling: process.env.VITE_USE_POLLING === "true" },
  },
});
