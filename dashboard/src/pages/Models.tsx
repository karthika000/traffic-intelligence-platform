import { useState } from "react";
import { useModels } from "../api/queries";
import type { ModelInfo } from "../api/types";
import { Empty, ErrorNote, Panel, Skeleton, Tag } from "../components/ui";
import { dateTime } from "../lib/format";
import { MLFLOW_URL } from "../lib/config";
import { keyMetrics } from "../lib/models";

type Run = { run_id: string; run_name?: string; end_time?: number; metrics?: Record<string, number> };
type Mlflow = { registered?: boolean; champion_version?: string | null; run_id?: string; description?: string; run?: Run; latest_run?: Run };


const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));
const modelUrl = (name: string, version?: string | null) =>
  `${MLFLOW_URL}/#/models/${encodeURIComponent(name)}${version ? `/versions/${encodeURIComponent(version)}` : ""}`;
const runUrl = (id: string) => `${MLFLOW_URL}/#/runs/${encodeURIComponent(id)}`;

function ModelCard({ m }: { m: ModelInfo }) {
  const [all, setAll] = useState(false);
  const ml = (m.mlflow ?? null) as Mlflow | null;
  const metrics = ml?.run?.metrics ?? {};
  const keyed = keyMetrics(metrics);
  const role = (m.details as { role?: string; loaded_by?: string } | undefined) ?? {};
  const champion = ml?.champion_version;
  const outOfDate = champion && champion !== m.version;
  return (
    <Panel
      title={
        <div className="min-w-0">
          <div className="font-semibold text-sm truncate">{m.model_name}</div>
          <div className="text-[11px] text-muted">
            {role.role ?? "model"}{role.loaded_by ? ` · loaded by ${role.loaded_by}` : ""}
          </div>
        </div>
      }
      actions={
        ml?.registered ? (
          <a className="text-xs link whitespace-nowrap" href={modelUrl(m.model_name, champion ?? m.version)} target="_blank" rel="noreferrer">
            MLflow ↗
          </a>
        ) : undefined
      }
    >
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <div className="text-[11px] text-faint">In service</div>
          <div className="font-semibold tabular-nums">v{m.version}</div>
          <div className="text-[11px] text-muted">{m.stage} · {dateTime(m.registered_at)}</div>
        </div>
        <div>
          <div className="text-[11px] text-faint">Champion</div>
          <div className="font-semibold tabular-nums">{champion ? `v${champion}` : "–"}</div>
          {outOfDate && <div className="text-[11px] text-[#fab219]">service is behind the champion</div>}
          {champion && !outOfDate && <div className="text-[11px] text-[#0ca30c]">in service</div>}
        </div>
      </div>
      {ml?.description && <p className="mt-3 text-xs text-muted">{ml.description}</p>}
      {!m.mlflow && <p className="mt-3 text-xs text-faint">Not tracked in MLflow (or MLflow unreachable).</p>}
      {keyed.length > 0 && (
        <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-2">
          {keyed.map(([k, v]) => (
            <div key={k} className="rounded-md bg-raised border border-line px-2 py-1.5 min-w-0">
              <div className="text-[10px] text-faint truncate" title={k}>{k}</div>
              <div className="font-semibold tabular-nums">{fmt(v)}</div>
            </div>
          ))}
        </div>
      )}
      {ml?.run && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-muted">
          champion run{" "}
          <a className="link" href={runUrl(ml.run.run_id)} target="_blank" rel="noreferrer">{ml.run.run_name ?? ml.run.run_id.slice(0, 8)}</a>
          {ml.latest_run && ml.latest_run.run_id !== ml.run.run_id && (
            <>
              · latest run{" "}
              <a className="link" href={runUrl(ml.latest_run.run_id)} target="_blank" rel="noreferrer">{ml.latest_run.run_name ?? ml.latest_run.run_id.slice(0, 8)}</a>
              {ml.latest_run.end_time && <span>({dateTime(new Date(ml.latest_run.end_time).toISOString())})</span>}
            </>
          )}
        </div>
      )}
      {Object.keys(metrics).length > keyed.length && (
        <div className="mt-2">
          <button className="text-[11px] link" onClick={() => setAll((a) => !a)}>
            {all ? "Hide" : "Show"} all {Object.keys(metrics).length} metrics
          </button>
          {all && (
            <table className="mt-1 w-full text-[11px] tabular-nums">
              <tbody>
                {Object.entries(metrics).sort().map(([k, v]) => (
                  <tr key={k} className="border-t border-line">
                    <td className="py-0.5 pr-2 text-muted break-all">{k}</td>
                    <td className="py-0.5 text-right">{fmt(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
      <div className="mt-3 pt-2 border-t border-line flex items-center gap-2 text-[11px] text-faint">
        <Tag>drift</Tag> status not available yet
      </div>
    </Panel>
  );
}

export function Models() {
  const q = useModels();
  return (
    <div className="flex flex-col gap-4">
      {q.isError && <ErrorNote error={q.error} onRetry={() => q.refetch()} />}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 items-start">
        {q.data
          ? q.data.map((m) => <ModelCard key={m.model_name} m={m} />)
          : !q.isError && [0, 1, 2].map((i) => <Skeleton key={i} className="h-64" />)}
      </div>
      {q.data?.length === 0 && <Empty>No models registered.</Empty>}
      <Panel title="Drift monitoring">
        <div className="text-sm text-muted">
          Drift status per model will appear here once <code className="text-xs">ml/monitoring</code> publishes to{" "}
          <code className="text-xs">drift.alerts</code>: feature and prediction drift scores, the reference window, and the last alert.
        </div>
      </Panel>
      <div className="text-xs text-faint">
        Model metrics come from MLflow at <a className="link" href={MLFLOW_URL} target="_blank" rel="noreferrer">{MLFLOW_URL}</a>, cached by the API for 60 s.
      </div>
    </div>
  );
}
