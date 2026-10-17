import { lazy, type ReactNode, Suspense } from "react";
import { Link, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Skeleton } from "./components/ui";
import { Overview } from "./pages/Overview";

// The overview loads with the app; other pages (and Recharts) load on first visit.
const CameraDetail = lazy(() => import("./pages/CameraDetail").then((m) => ({ default: m.CameraDetail })));
const Demo = lazy(() => import("./pages/Demo").then((m) => ({ default: m.Demo })));
const IncidentDetail = lazy(() => import("./pages/IncidentDetail").then((m) => ({ default: m.IncidentDetail })));
const Incidents = lazy(() => import("./pages/Incidents").then((m) => ({ default: m.Incidents })));
const Models = lazy(() => import("./pages/Models").then((m) => ({ default: m.Models })));
const Reports = lazy(() => import("./pages/Reports").then((m) => ({ default: m.Reports })));

const page = (el: ReactNode) => <Suspense fallback={<Skeleton className="h-96" />}>{el}</Suspense>;

function NotFound() {
  return (
    <div className="py-16 text-center">
      <p className="text-muted">Page not found.</p>
      <Link to="/" className="link text-sm">Back to the overview</Link>
    </div>
  );
}

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="cameras/:id" element={page(<CameraDetail />)} />
        <Route path="incidents" element={page(<Incidents />)} />
        <Route path="incidents/:id" element={page(<IncidentDetail />)} />
        <Route path="reports" element={page(<Reports />)} />
        <Route path="models" element={page(<Models />)} />
        <Route path="demo" element={page(<Demo />)} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
