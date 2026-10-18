import { useState } from "react";
import { useDriftHistory, useDriftStatus, useModels } from "../api/queries";
import type { DriftHistoryPoint, DriftStatus, ModelInfo } from "../api/types";
import { DriftModeControl, DriftPill, DriftSection } from "../components/Drift";
import { Empty, ErrorNote, Panel, Segmented, Skeleton, Tag } from "../components/ui";
import { MONITORED_MODELS } from "../lib/drift";
import { dateTime } from "../lib/format";
import { MLFLOW_URL } from "../lib/config";
import { keyMetrics } from "../lib/models";

type Run = { run_id: string; run_name?: string; end_time?: number; metrics?: Record<string, number> };
type Mlflow = { registered?: boolean; champion_version?: string | null; run_id?: string; description?: string; run?: Run; latest_run?: Run };


const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));
const modelUrl = (name: string, version?: string | null) =>
  `${MLFLOW_URL}/#/models/${encodeURIComponent(name)}${version ? `/versions/${encodeURIComponent(version)}` : ""}`;
const runUrl = (id: string) => `${MLFLOW_URL}/#/runs/${encodeURIComponent(id)}`;

type DriftProps = { drift?: DriftStatus; history: DriftHistoryPoint[]; hours: number };

function ModelCard({ m, drift, history, hours }: { m: ModelInfo } & DriftProps) {
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
      {(MONITORED_MODELS as readonly string[]).includes(m.model_name) ? (
        <DriftSection status={drift} history={history} hours={hours} />
      ) : (
        <div className="mt-3 pt-2 border-t border-line flex items-center gap-2 text-[11px] text-faint">
          <Tag>drift</Tag> not monitored (no drift checks for this model)
        </div>
      )}
    </Panel>
  );
}

type Range = "6" | "24" | "72";
const RANGES: { value: Range; label: string }[] = [
  { value: "6", label: "6 h" },
  { value: "24", label: "24 h" },
  { value: "72", label: "3 d" },
];

export function Models() {
  const q = useModels();
  const [range, setRange] = useState<Range>("6");
  const hours = Number(range);
  const drift = useDriftStatus();
  const history = useDriftHistory(hours);
  const byModel = new Map((drift.data ?? []).map((d) => [d.model_name, d]));
  const historyOf = (name: string) => (history.data ?? []).filter((p) => p.model_name === name);
  const order = (m: ModelInfo) => {
    const i = (MONITORED_MODELS as readonly string[]).indexOf(m.model_name);
    return i < 0 ? MONITORED_MODELS.length : i;
  };
  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="Drift monitoring"
        actions={<Segmented value={range} onChange={setRange} options={RANGES} label="Drift history range" />}
      >
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-4 items-start">
          <div className="flex flex-col gap-2 text-sm">
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              {(drift.data ?? []).map((d) => (
                <span key={d.model_name} className="inline-flex items-center gap-1.5">
                  <span className="text-muted">{d.model_name}</span>
                  <DriftPill status={d.status} />
                </span>
              ))}
            </div>
            <p className="text-xs text-faint">
              ml-monitor compares the last window of production data with each champion's reference snapshot (Evidently): a feature drifts when its
              score reaches its threshold, a model is <b>drifted</b> when the share of drifted features reaches the dashed line, and a{" "}
              <b>warning</b> when fewer features drifted or a data quality check failed. Drift alerts record a retraining recommendation;
              retraining is gated on beating the champion.
            </p>
            {drift.isError && <ErrorNote error={drift.error} onRetry={() => drift.refetch()} compact />}
          </div>
          <div className="lg:border-l lg:border-line lg:pl-4">
            <DriftModeControl />
          </div>
        </div>
      </Panel>
      {q.isError && <ErrorNote error={q.error} onRetry={() => q.refetch()} />}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 items-start">
        {q.data
          ? [...q.data]
              .sort((a, b) => order(a) - order(b))
              .map((m) => (
                <ModelCard key={m.model_name} m={m} drift={byModel.get(m.model_name)} history={historyOf(m.model_name)} hours={hours} />
              ))
          : !q.isError && [0, 1, 2].map((i) => <Skeleton key={i} className="h-64" />)}
      </div>
      {q.data?.length === 0 && <Empty>No models registered.</Empty>}
      <div className="text-xs text-faint">
        Model metrics come from MLflow at <a className="link" href={MLFLOW_URL} target="_blank" rel="noreferrer">{MLFLOW_URL}</a>, cached by the API for 60 s.
      </div>
    </div>
  );
}
