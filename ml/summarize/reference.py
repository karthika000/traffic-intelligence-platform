"""Reference summaries for evaluation: written from the same facts, but in a different
operator-briefing style from the template (so the template isn't scored against itself),
plus ROUGE-L."""

import re

from ml.summarize.context import IncidentContext

VERB = {"suspected": "is suspected", "confirmed": "is confirmed", "resolved": "has cleared"}


def reference_headline(ctx: IncidentContext) -> str:
    state = {"suspected": "possible", "confirmed": "ongoing", "resolved": "now clear"}[ctx.status]
    return f"{ctx.type_label.capitalize()} at {ctx.short_location}: {state}, {ctx.severity} impact"


def reference_summary(ctx: IncidentContext) -> str:
    out = [
        f"A {ctx.severity}-severity {ctx.type_label} {VERB.get(ctx.status, ctx.status)} at "
        f"{ctx.short_location}, first seen at {ctx.start_local}."
    ]
    impact = []
    if ctx.loop_pct is not None:
        impact.append(
            f"vehicles are moving at {ctx.loop_speed_kmh:.0f} km/h instead of the usual "
            f"{ctx.loop_baseline_kmh:.0f} km/h"
        )
    elif ctx.speed_pct is not None:
        impact.append(
            f"vehicles are moving at {ctx.speed_kmh:.0f} km/h instead of the usual "
            f"{ctx.baseline_speed_kmh:.0f} km/h"
        )
    if ctx.stopped_vehicles:
        impact.append(f"with {ctx.stopped_vehicles} stopped")
    if ctx.congestion:
        impact.insert(0, f"traffic is {ctx.congestion.replace('_', ' ')}")
    if impact:
        out.append(" ".join(impact).capitalize().replace(" with", ", with", 1) + ".")
    basis = []
    if ctx.n_reports:
        basis.append(f"{ctx.n_reports} reports from {', '.join(ctx.source_labels)}")
    if ctx.vision_present:
        basis.append("the camera feed")
    if basis:
        out.append(f"This is based on {' and '.join(basis)}.")
    out.append(f"Operators should {ctx.recommended_action}.")
    return " ".join(out)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1 over lowercase word tokens (longest common subsequence)."""
    c, r = _tokens(candidate), _tokens(reference)
    if not c or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for x in c:
        cur = [0]
        for j, y in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(c), lcs / len(r)
    return 2 * precision * recall / (precision + recall)
