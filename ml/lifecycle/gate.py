"""The promotion gate: does a candidate replace the champion?

Both are evaluated on the same held-out data. The candidate is promoted only if

1. it beats the champion on the **primary** metric by more than ``min_improvement``, and
2. it does not regress on any **guard** metric by more than that metric's tolerance (or the
   policy's default tolerance).

Without a champion the candidate is promoted. Metrics can be higher- or lower-is-better;
tolerances are absolute, in the metric's own units.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool = True
    tolerance: float | None = None  # None: the policy's default tolerance
    label: str = ""


@dataclass(frozen=True)
class GatePolicy:
    primary: MetricSpec
    guards: tuple[MetricSpec, ...] = ()
    tolerance: float = 0.01
    min_improvement: float = 0.0


@dataclass
class Decision:
    promoted: bool
    reasons: list[str]
    comparison: list[dict[str, Any]] = field(default_factory=list)
    primary_metric: str = ""
    primary_delta: float | None = None

    @property
    def decision(self) -> str:
        return "promoted" if self.promoted else "rejected"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "decision": self.decision}


def _value(metrics: dict[str, float] | None, name: str) -> float | None:
    if not metrics or name not in metrics or metrics[name] is None:
        return None
    v = float(metrics[name])
    return v if math.isfinite(v) else None


def decide(
    policy: GatePolicy,
    champion: dict[str, float] | None,
    candidate: dict[str, float],
) -> Decision:
    reasons: list[str] = []
    comparison: list[dict[str, Any]] = []
    p = policy.primary
    cand = _value(candidate, p.name)
    if cand is None:
        return Decision(False, [f"candidate has no {p.name}"], [], p.name)
    champ = _value(champion, p.name)
    if champ is None:
        comparison.append({"metric": p.name, "role": "primary", "champion": None,
                           "candidate": cand, "delta": None, "ok": True})  # fmt: skip
        return Decision(True, [f"no champion to compare with; {p.name} {cand:.4f}"],
                        comparison, p.name)  # fmt: skip

    sign = 1.0 if p.higher_is_better else -1.0
    improvement = (cand - champ) * sign
    primary_ok = improvement > policy.min_improvement
    comparison.append({"metric": p.name, "role": "primary", "champion": champ, "candidate": cand,
                       "delta": cand - champ, "ok": primary_ok})  # fmt: skip
    if primary_ok:
        reasons.append(f"beats the champion on {p.name}: {cand:.4f} vs {champ:.4f} "
                       f"({cand - champ:+.4f})")  # fmt: skip
    else:
        need = (
            f"more than {policy.min_improvement:g} better" if policy.min_improvement else "better"
        )
        reasons.append(f"does not beat the champion on {p.name}: {cand:.4f} vs {champ:.4f} "
                       f"({cand - champ:+.4f}; needs to be {need})")  # fmt: skip

    regressions = []
    for g in policy.guards:
        c, h = _value(candidate, g.name), _value(champion, g.name)
        tolerance = policy.tolerance if g.tolerance is None else g.tolerance
        if c is None or h is None:
            comparison.append({"metric": g.name, "role": "guard", "champion": h, "candidate": c,
                               "delta": None, "tolerance": tolerance, "ok": True,
                               "note": "not available for both"})  # fmt: skip
            continue
        regression = (h - c) * (1.0 if g.higher_is_better else -1.0)
        ok = regression <= tolerance + 1e-12
        comparison.append({"metric": g.name, "role": "guard", "champion": h, "candidate": c,
                           "delta": c - h, "tolerance": tolerance, "ok": ok})  # fmt: skip
        if not ok:
            regressions.append(
                f"regresses on {g.name}: {c:.4f} vs {h:.4f} ({c - h:+.4f}, tolerance {tolerance:g})"
            )
    reasons.extend(regressions)
    if primary_ok and not regressions:
        reasons.append(
            f"no guard metric regressed beyond its tolerance ({len(policy.guards)} checked)"
        )
    return Decision(primary_ok and not regressions, reasons, comparison, p.name, cand - champ)
