"""Compare incident summarizer configurations and register the chosen one.

    docker compose run --rm ml-trainer python -m ml.evaluation.summarizer_eval

Reference set (~40 contexts): the stored incidents (facts from Postgres), topped up with
contexts built from scenario ground truth (every phase of every scenario run: its type, peak
severity so far, status, and loop-detector speeds from the phase's speed factor; the camera,
whose looped footage doesn't show simulated incidents, stays at its usual speed). On a fresh
install, with no completed runs yet, the same contexts are built from the scenario definitions
(``--scenarios-dir``) as if each scenario had run once on each camera. With
``--reference-set`` (default data/summarizer/reference_set.json, when present) exactly those
contexts are used, so runs stay comparable. Reference summaries are written from the same facts
in a different style (ml.summarize.reference).

Configurations (``--models``, "name:prompt[:architecture]"): the deterministic template,
flan-t5-small / flan-t5-base (seq2seq, prompt v1) and Qwen2.5-Instruct chat models (prompt v2).
Each model is first timed on ``--probe`` contexts; one whose p95 exceeds ``--max-p95`` seconds
is excluded. Per configuration: ROUGE-L of the raw model output and of what the service would
publish (model output, or the template where the faithfulness guard rejects it), the guard's
pass rate, the fallback rate, and CPU latency (threads capped like the service).

The champion is the configuration with the best published ROUGE-L among those with a guard
pass rate of at least ``--min-pass`` (0.9) that fit the service's time budget (p95); a tie
(within 0.005) goes to the cheaper one. It is registered as "incident-summarizer" @champion.
Everything is logged to MLflow experiment "incident-summarizer".
"""

import argparse
import json
import os
import re
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import text

from ml.summarize import faithfulness
from ml.summarize.context import IncidentContext, load_context
from ml.summarize.pyfunc import IncidentSummarizerModel, write_config
from ml.summarize.reference import reference_headline, reference_summary, rouge_l
from ml.summarize.summarizer import SummarizerConfig, tidy
from ml.summarize.template import template_headline, template_summary
from ml.training.common import ML_DIR, log, pip_requirements, register, setup
from traffic_common.db import create_db_engine

EXPERIMENT = MODEL_NAME = "incident-summarizer"
SEVERITIES = ["low", "medium", "high", "critical"]
SOURCE_MAP = {
    "driver": "citizen",
    "social_media": "social_media",
    "police": "police",
    "operator": "operator",
}
TYPE_VEHICLES = {"accident": ["car", "bus"], "breakdown": ["lorry"], "debris": ["van"]}


def stored_contexts(conn, limit: int) -> list[IncidentContext]:
    ids = conn.scalars(
        text("SELECT id FROM incidents ORDER BY updated_at DESC LIMIT :n"), {"n": limit}
    ).all()
    out = [load_context(conn, i) for i in ids]
    return [c for c in out if c is not None and c.incident_type != "none"]


def scenario_contexts(conn) -> list[IncidentContext]:
    runs = conn.execute(
        text(
            "SELECT r.id, r.camera_id, r.incident_type, r.started_at, r.timeline, r.definition, "
            "c.name, c.location_name FROM scenario_runs r JOIN cameras c ON c.id = r.camera_id "
            "WHERE r.status = 'completed' ORDER BY r.started_at"
        )
    ).all()
    return [ctx for run in runs for ctx in run_contexts(run)]


def _seconds(value) -> float:
    """A scenario duration: seconds, or "90s" / "2m" / "1h"."""
    if isinstance(value, int | float):
        return float(value)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", str(value))
    if not m:
        raise ValueError(f"bad duration {value!r}")
    return float(m[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[m[2]]


def definition_contexts(conn, scenarios_dir: Path) -> list[IncidentContext]:
    """Contexts as scenario_contexts() would build them, from the scenario definitions instead
    of completed runs: each scenario run once on each camera, at fixed times so the set is the
    same on every fresh install."""
    cameras = conn.execute(text("SELECT id, name, location_name FROM cameras ORDER BY id")).all()
    files = sorted(scenarios_dir.glob("*.yaml"))
    if not files:
        raise SystemExit(f"no scenario definitions in {scenarios_dir.resolve()}")
    out, start = [], datetime(2026, 1, 15, 7, 0, tzinfo=UTC)
    for path in files:
        definition = yaml.safe_load(path.read_text(encoding="utf-8"))
        for phase in definition["phases"]:
            phase["duration"] = _seconds(phase["duration"])
        for camera in cameras:
            start += timedelta(hours=1)
            t, timeline = start, []
            for phase in definition["phases"]:
                t += timedelta(seconds=phase["duration"])
                timeline.append({"ends_at": t.isoformat()})
            run = SimpleNamespace(
                id=f"definition-{definition['name']}-{camera.id}",
                camera_id=camera.id,
                incident_type=definition["incident_type"],
                started_at=start,
                timeline=timeline,
                definition=definition,
                name=camera.name,
                location_name=camera.location_name,
            )
            out += run_contexts(run)
    return out


def run_contexts(run) -> list[IncidentContext]:
    """One context per phase of a scenario run, with the facts as of the end of that phase."""
    out = []
    phases = run.definition.get("phases", [])
    timeline = run.timeline or []
    peak, reports, sources = "low", 0, set()
    for i, (phase, actual) in enumerate(zip(phases, timeline, strict=False)):
        peak = max(peak, phase["severity"], key=SEVERITIES.index)
        per_min = (phase.get("reports") or {}).get("per_minute", 0)
        reports += round(per_min * phase["duration"] / 60)
        sources |= {SOURCE_MAP.get(s, s) for s in (phase.get("reports") or {}).get("sources", {})}
        factor = (phase.get("traffic") or {}).get("speed_factor", 1.0)
        status = "suspected" if i == 0 else "resolved" if i == len(phases) - 1 else "confirmed"
        congestion = (
            "standstill"
            if factor < 0.25
            else "heavy"
            if factor < 0.5
            else "moderate"
            if factor < 0.8
            else "free_flow"
        )
        start = pd.Timestamp(run.started_at).to_pydatetime()
        out.append(
            IncidentContext(
                incident_id=f"{run.id}:{phase['name']}",
                version=i + 1,
                camera_id=run.camera_id,
                location=run.name,
                area=run.location_name,
                incident_type=run.incident_type,
                severity=peak,
                status=status,
                start_time=start,
                reference_time=pd.Timestamp(actual["ends_at"]).to_pydatetime(),
                congestion=congestion,
                # Loops see the incident; the looped camera footage doesn't.
                speed_kmh=25.0,
                baseline_speed_kmh=25.0,
                loop_speed_kmh=round(30 * factor, 1),
                loop_baseline_kmh=30.0,
                stopped_vehicles=max(0, round(4 * (1 - factor) - 1)),
                n_reports=reports,
                sources=sorted(sources),
                vehicles=TYPE_VEHICLES.get(run.incident_type, []),
                fused_confidence=round(0.55 + 0.4 * (1 - factor), 2),
                vision_present=factor < 0.8,
            )
        )
    return out


def build_dataset(
    engine, size: int, pinned: list[str] | None = None, scenarios_dir: Path | None = None
) -> list[IncidentContext]:
    with engine.connect() as conn:
        synthetic = scenario_contexts(conn)
        if not synthetic and not pinned and scenarios_dir is not None:
            synthetic = definition_contexts(conn, scenarios_dir)
            log.info("no completed scenario runs: %d contexts from the definitions", len(synthetic))
        if pinned:
            by_id = {c.incident_id: c for c in synthetic}
            out = []
            for key in pinned:
                ctx = by_id.get(key) if ":" in key else load_context(conn, key)
                if ctx is None:
                    raise SystemExit(f"reference set item {key} no longer exists")
                out.append(ctx)
            return out
        stored = stored_contexts(conn, size // 2)
    rng = np.random.default_rng(7)
    rng.shuffle(synthetic)
    return (stored + synthetic)[:size]


def parse_models(spec: str, timeout: float) -> list[tuple[SummarizerConfig, bool]]:
    """'google/flan-t5-base:v1,Qwen/Qwen2.5-1.5B-Instruct:v2:chat?' -> (configuration, gated).
    A trailing "?" means: only evaluate it if the latency probe's p95 is under --max-p95."""
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        gated = item.endswith("?")
        name, prompt, *arch = item.rstrip("?").split(":")
        config = SummarizerConfig(
            model_name=name,
            prompt=prompt,
            architecture=arch[0] if arch else "seq2seq",
            timeout_seconds=timeout,
        )
        out.append((config, gated))
    return out


# ----------------------------------------------------------------------------- running


def run_template(ctxs: list[IncidentContext]) -> list[dict]:
    rows = []
    for ctx in ctxs:
        started = time.perf_counter()
        h, s = template_headline(ctx), template_summary(ctx)
        ms = (time.perf_counter() - started) * 1000
        verdict = faithfulness.check(f"{h}\n{s}", ctx)
        rows.append(
            dict(
                raw_headline=h,
                raw_summary=s,
                headline=h,
                summary=s,
                latency_ms=ms,
                passed=verdict.passed,
                violations=verdict.violations,
                fallback=False,
            )
        )
    return rows


def run_model(
    config: SummarizerConfig, ctxs: list[IncidentContext], timeout: float, summarizer=None
) -> list[dict]:
    """Exactly what the service does (IncidentSummarizer): the published output is the model's
    when it passes the guard and the completeness checks, the template otherwise."""
    from ml.summarize.summarizer import IncidentSummarizer

    if summarizer is None:
        summarizer = IncidentSummarizer(config)
        summarizer.generator.load()
        summarizer.summarize(ctxs[0], 120)  # warm-up, not timed
    rows = []
    for ctx in ctxs:
        started = time.perf_counter()
        r = summarizer.summarize(ctx, timeout)
        ms = (time.perf_counter() - started) * 1000
        f = r.faithfulness
        if r.generation == "model":
            raw_h, raw_s, passed, violations = r.headline, r.summary, True, []
        else:
            out = f.get("model_output") or {"headline": "", "summary": ""}
            raw_h, raw_s = out["headline"], out["summary"]
            h, s, _ = tidy(raw_h, raw_s, ctx)
            verdict = faithfulness.check(h + " " + s, ctx) if raw_s else None
            passed = bool(verdict and verdict.passed)
            violations = verdict.violations if verdict else [f.get("fallback_reason")]
        rows.append(
            dict(
                raw_headline=raw_h,
                raw_summary=raw_s,
                headline=r.headline,
                summary=r.summary,
                latency_ms=ms,
                passed=passed,
                violations=violations,
                fallback=r.generation == "template",
                fallback_reason=f.get("fallback_reason"),
            )
        )
    return rows


def score(rows: list[dict], ctxs: list[IncidentContext]) -> dict[str, float]:
    raw, published = [], []
    for row, ctx in zip(rows, ctxs, strict=True):
        ref = f"{reference_headline(ctx)} {reference_summary(ctx)}"
        raw.append(rouge_l(f"{row['raw_headline']} {row['raw_summary']}", ref))
        published.append(rouge_l(f"{row['headline']} {row['summary']}", ref))
    lat = [r["latency_ms"] for r in rows]
    return {
        "rouge_l_raw": float(np.mean(raw)),
        "rouge_l_published": float(np.mean(published)),
        "faithfulness_pass_rate": float(np.mean([r["passed"] for r in rows])),
        "fallback_rate": float(np.mean([r["fallback"] for r in rows])),
        # faithful, but rejected as unusable (cut-off headline, type or place missing)
        "incomplete_rate": float(
            np.mean(
                [
                    str(r.get("fallback_reason") or "").startswith("model output incomplete")
                    for r in rows
                ]
            )
        ),
        "latency_mean_ms": float(np.mean(lat)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "raw_words_mean": float(np.mean([len(r["raw_summary"].split()) for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument(
        "--models",
        default="google/flan-t5-small:v1,google/flan-t5-base:v1,"
        "Qwen/Qwen2.5-0.5B-Instruct:v2:chat,Qwen/Qwen2.5-1.5B-Instruct:v2:chat?",
    )
    parser.add_argument(
        "--reference-set", type=Path, default=Path("data/summarizer/reference_set.json")
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=Path(os.getenv("SCENARIOS_DIR", "scenarios")),
        help="scenario YAML files, used when no scenario run has completed yet",
    )
    parser.add_argument("--probe", type=int, default=5, help="contexts timed before a full run")
    parser.add_argument("--max-p95", type=float, default=10.0, help="seconds, from the probe")
    parser.add_argument("--min-pass", type=float, default=0.9, help="guard pass rate to promote")
    parser.add_argument("--threads", type=int, default=2, help="like the service")
    parser.add_argument("--timeout", type=float, default=20.0, help="the service's budget")
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()
    setup(args.threads)
    import mlflow

    engine = create_db_engine(os.getenv("POSTGRES_URL"))
    pinned = None
    if args.reference_set and args.reference_set.exists():
        pinned = json.loads(args.reference_set.read_text())["incident_ids"]
        log.info("using the pinned reference set %s", args.reference_set)
    ctxs = build_dataset(engine, args.size, pinned, args.scenarios_dir)
    if not ctxs:
        raise SystemExit("no contexts to evaluate: no incidents, scenario runs or definitions")
    log.info(
        "reference set: %d contexts (%d stored incidents)",
        len(ctxs),
        sum(":" not in c.incident_id for c in ctxs),
    )
    work = Path(tempfile.mkdtemp(prefix="summarizer-eval-"))
    with (work / "dataset.jsonl").open("w", encoding="utf-8") as f:
        for ctx in ctxs:
            f.write(
                json.dumps(
                    {
                        "context": ctx.to_dict(),
                        "facts": ctx.facts_text(),
                        "reference_headline": reference_headline(ctx),
                        "reference_summary": reference_summary(ctx),
                    },
                    default=str,
                )
                + "\n"
            )

    configs = [(SummarizerConfig(generator="template", model_name="template", prompt="-"), False)]
    configs += parse_models(args.models, args.timeout)
    results, excluded = [], []
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"summarizer-eval-{time.strftime('%Y%m%d-%H%M%S')}") as parent:
        mlflow.log_params(
            {
                "contexts": len(ctxs),
                "threads": args.threads,
                "timeout_s": args.timeout,
                "models": args.models,
                "reference_set": str(pinned and args.reference_set),
                "min_pass": args.min_pass,
                "max_p95_s": args.max_p95,
            }
        )
        mlflow.log_artifact(str(work / "dataset.jsonl"), artifact_path="dataset")
        for config, gated in configs:
            label = (
                "template" if config.generator == "template" else config.model_name.split("/")[-1]
            )
            with mlflow.start_run(run_name=label, nested=True) as run:
                mlflow.log_params({f"config_{k}": v for k, v in config.to_dict().items()})
                started = time.perf_counter()
                if config.generator == "template":
                    rows = run_template(ctxs)
                else:
                    from ml.summarize.summarizer import IncidentSummarizer

                    summarizer = IncidentSummarizer(config)
                    summarizer.generator.load()
                    summarizer.summarize(ctxs[0], 120)  # warm-up, not timed
                    # Latency probe with a generous budget: is it worth a full run?
                    probe = run_model(config, ctxs[: args.probe], 120.0, summarizer)
                    p95 = float(np.percentile([r["latency_ms"] for r in probe], 95)) / 1000
                    mlflow.log_metric("probe_latency_p95_s", p95)
                    if gated and p95 > args.max_p95:
                        mlflow.set_tag("excluded", f"probe p95 {p95:.1f}s > {args.max_p95:g}s")
                        log.info("%s excluded: probe p95 %.1fs", label, p95)
                        excluded.append({"configuration": label, "probe_p95_s": round(p95, 1)})
                        continue
                    rows = run_model(config, ctxs, args.timeout, summarizer)
                metrics = score(rows, ctxs)
                metrics["eval_seconds"] = time.perf_counter() - started
                mlflow.log_metrics(metrics)
                out = pd.DataFrame(rows).assign(
                    incident=[c.incident_id for c in ctxs],
                    reference=[reference_summary(c) for c in ctxs],
                )
                out.to_csv(work / f"outputs_{label}.csv", index=False)
                mlflow.log_artifact(str(work / f"outputs_{label}.csv"), artifact_path="outputs")
                results.append(
                    {"configuration": label, "config": config, "run_id": run.info.run_id, **metrics}
                )
                log.info("%s: %s", label, {k: round(v, 3) for k, v in metrics.items()})

        table = pd.DataFrame(
            [{k: v for k, v in r.items() if k not in ("config", "run_id")} for r in results]
        )
        fits = [
            r
            for r in results
            if r["latency_p95_ms"] <= args.timeout * 1000
            and r["faithfulness_pass_rate"] >= args.min_pass
        ]
        best = max(fits, key=lambda r: r["rouge_l_published"])
        cheaper = [
            r
            for r in fits
            if r["latency_mean_ms"] < best["latency_mean_ms"]
            and best["rouge_l_published"] - r["rouge_l_published"] <= 0.005
        ]
        chosen = min(cheaper, key=lambda r: r["latency_mean_ms"]) if cheaper else best
        rationale = (
            f"best published ROUGE-L with guard pass rate >= {args.min_pass:g} within the "
            f"{args.timeout:g}s budget" + (" (cheaper config within 0.005)" if cheaper else "")
        )
        table.to_csv(work / "comparison.csv", index=False)
        mlflow.log_artifact(str(work / "comparison.csv"), artifact_path="comparison")
        mlflow.set_tags({"chosen": chosen["configuration"], "rationale": rationale})

        version = None
        if not args.no_promote:
            config = chosen["config"]
            config.timeout_seconds = args.timeout
            config_path = write_config(config, work / "model" / "config.json")
            with mlflow.start_run(run_id=chosen["run_id"], nested=True):
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=IncidentSummarizerModel(),
                    artifacts={"config": str(config_path)},
                    code_paths=[str(ML_DIR)],
                    pip_requirements=pip_requirements(
                        "requirements-runtime.txt", "requirements-nlp.txt"
                    ),
                )
            from ml.summarize.prompts import PromptTemplate

            prompt_version = (
                PromptTemplate.load(config.prompt).version if config.generator == "model" else "-"
            )
            version = register(
                chosen["run_id"],
                MODEL_NAME,
                "champion",
                f"{chosen['configuration']} (prompt {prompt_version}): published ROUGE-L "
                f"{chosen['rouge_l_published']:.3f}, guard pass rate "
                f"{chosen['faithfulness_pass_rate']:.2f}; {rationale}",
                {
                    "generator": config.generator,
                    "model_name": config.model_name,
                    "architecture": config.architecture,
                    "prompt_version": prompt_version,
                    "parent_run": parent.info.run_id,
                },
            )
            mlflow.set_tag("registered_version", version)

    pd.set_option("display.width", 200)
    print("\n" + table.round(3).to_string(index=False))
    for e in excluded:
        print(f"excluded: {e['configuration']} (probe p95 {e['probe_p95_s']}s > {args.max_p95:g}s)")
    print(
        json.dumps(
            {
                "chosen": chosen["configuration"],
                "rationale": rationale,
                "registered_version": version,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
