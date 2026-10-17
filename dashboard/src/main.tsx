import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "leaflet/dist/leaflet.css";
import "./index.css";
import { ApiError } from "./api/client";
import { App } from "./App";
import { LiveProvider } from "./live/LiveProvider";
import { applyTheme, themeStore } from "./lib/theme";

applyTheme(themeStore.getState().theme);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Don't hammer an API that is down: 4xx never retries, other failures twice with backoff.
      retry: (count, err) => !(err instanceof ApiError && err.status >= 400 && err.status < 500) && count < 2,
      retryDelay: (n) => Math.min(8000, 1000 * 2 ** n),
      refetchOnWindowFocus: false,
      staleTime: 5_000,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <LiveProvider>
          <App />
        </LiveProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
