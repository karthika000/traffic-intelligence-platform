"""Deterministic summary from the facts alone: the fallback whenever the model's output can't
be used, and the "template" configuration in the evaluation. Faithful by construction."""

from ml.summarize.context import IncidentContext

STATUS_PHRASE = {
    "suspected": "Suspected",
    "confirmed": "Confirmed",
    "resolved": "Resolved",
}


def template_headline(ctx: IncidentContext) -> str:
    status = {"suspected": "suspected", "confirmed": "confirmed", "resolved": "cleared"}
    words = f"{ctx.severity.capitalize()} {ctx.type_label} at {ctx.short_location}, " + status.get(
        ctx.status, ctx.status
    )
    return " ".join(words.split()[:14])


def template_summary(ctx: IncidentContext) -> str:
    sentences = []
    what = (
        f"{STATUS_PHRASE.get(ctx.status, ctx.status.capitalize())} {ctx.type_label} at "
        f"{ctx.location}, severity {ctx.severity}, since {ctx.start_local}"
    )
    if ctx.status == "resolved":
        what += f" (lasted {ctx.duration_min} minutes)"
    else:
        what += f" ({ctx.duration_min} minutes)"
    sentences.append(what + ".")

    impact = []
    if ctx.congestion:
        impact.append(f"congestion is {ctx.congestion.replace('_', ' ')}")
    if ctx.loop_pct is not None:
        impact.append(
            f"loop detectors measure {ctx.loop_speed_kmh:.0f} km/h against "
            f"{ctx.loop_baseline_kmh:.0f} km/h normally"
        )
    if ctx.show_camera_speed:
        impact.append(
            f"camera speeds are {ctx.speed_kmh:.0f} km/h against {ctx.baseline_speed_kmh:.0f} "
            f"km/h normally"
        )
    if ctx.stopped_vehicles:
        impact.append(f"{ctx.stopped_vehicles} vehicles are stopped")
    if ctx.pedestrians_in_roadway:
        impact.append(f"{ctx.pedestrians_in_roadway} pedestrians are in the roadway")
    if ctx.lanes:
        impact.append(f"reports mention {', '.join(ctx.lanes)}")
    if impact:
        sentences.append(_cap("; ".join(impact)) + ".")

    evidence = []
    if ctx.n_reports:
        evidence.append(
            f"{ctx.n_reports} reports from {len(ctx.sources)} sources "
            f"({', '.join(ctx.source_labels)})"
        )
    if ctx.vehicles:
        evidence.append(f"vehicles mentioned: {', '.join(ctx.vehicles)}")
    if ctx.vision_present:
        evidence.append(
            "a traffic anomaly on the camera"
            + (" (vision degraded)" if ctx.vision_degraded else "")
        )
    if evidence:
        sentences.append(f"Evidence: {'; '.join(evidence)}; confidence {ctx.fused_confidence:.2f}.")
    sentences.append(f"Recommended action: {ctx.recommended_action}.")
    return " ".join(sentences[:4])


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]
