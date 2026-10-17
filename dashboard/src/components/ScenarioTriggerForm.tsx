import { type FormEvent, useEffect, useState } from "react";
import type { CameraStatus, ScenarioInfo, TriggerResponse } from "../api/types";
import { ApiError } from "../api/client";
import { titleCase } from "../lib/format";
import { Field } from "./ui";

export const TIME_SCALES = [1, 2, 6, 10, 20] as const;

export type TriggerInput = { scenario: string; camera_id: string; time_scale: number };

const minutes = (s: number) => (s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`);

export function ScenarioTriggerForm({
  scenarios,
  cameras,
  onTrigger,
  defaults,
}: {
  scenarios: ScenarioInfo[];
  cameras: Pick<CameraStatus, "id" | "name">[];
  onTrigger: (input: TriggerInput) => Promise<TriggerResponse>;
  defaults?: Partial<TriggerInput>;
}) {
  const [scenario, setScenario] = useState(defaults?.scenario ?? "");
  const [camera, setCamera] = useState(defaults?.camera_id ?? "");
  const [timeScale, setTimeScale] = useState(defaults?.time_scale ?? 6);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<{ ok: true; res: TriggerResponse } | { ok: false; message: string } | null>(null);

  // Pick the first options once they load, unless the user already chose.
  useEffect(() => {
    if (!scenario && scenarios.length) setScenario(scenarios[0].name);
  }, [scenarios, scenario]);
  useEffect(() => {
    if (!camera && cameras.length) setCamera(cameras[0].id);
  }, [cameras, camera]);

  const info = scenarios.find((s) => s.name === scenario);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!scenario || !camera || pending) return;
    setPending(true);
    setResult(null);
    try {
      const res = await onTrigger({ scenario, camera_id: camera, time_scale: timeScale });
      setResult({ ok: true, res });
    } catch (err) {
      setResult({ ok: false, message: err instanceof ApiError || err instanceof Error ? err.message : "Trigger failed" });
    } finally {
      setPending(false);
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-3" aria-label="Trigger a scenario">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <Field label="Scenario">
          <select className="input" value={scenario} onChange={(e) => setScenario(e.target.value)} aria-label="Scenario">
            {scenarios.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Camera">
          <select className="input" value={camera} onChange={(e) => setCamera(e.target.value)} aria-label="Camera">
            {cameras.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} ({c.id})
              </option>
            ))}
          </select>
        </Field>
        <Field label="Time scale">
          <select className="input" value={timeScale} onChange={(e) => setTimeScale(Number(e.target.value))} aria-label="Time scale">
            {TIME_SCALES.map((t) => (
              <option key={t} value={t}>
                ×{t} {t === 1 ? "(real time)" : "faster"}
              </option>
            ))}
          </select>
        </Field>
      </div>
      {info && (
        <div className="text-xs text-muted rounded-md bg-raised border border-line p-2.5">
          <div className="mb-1.5">{info.description}</div>
          <div className="flex flex-wrap gap-1.5 items-center">
            <span className="font-medium text-ink">{titleCase(info.incident_type)}</span>
            <span>· {minutes(info.duration_seconds / timeScale)} at ×{timeScale}:</span>
            {info.phases.map((p) => (
              <span key={p.name} className="rounded bg-panel border border-line px-1.5 py-0.5">
                {p.name} {minutes(p.duration_seconds / timeScale)}
              </span>
            ))}
          </div>
        </div>
      )}
      <div className="flex items-center gap-3">
        <button type="submit" className="btn-primary" disabled={!scenario || !camera || pending}>
          {pending ? "Triggering…" : "Trigger scenario"}
        </button>
        {result?.ok && (
          <span className="text-sm" role="status">
            Started <code className="text-xs">{result.res.scenario_id.slice(0, 8)}</code> ({result.res.status ?? "accepted"})
          </span>
        )}
        {result && !result.ok && (
          <span className="text-sm text-[#ee4b26]" role="alert">
            {result.message}
          </span>
        )}
      </div>
    </form>
  );
}
