"""``bioguard eval`` -- replay the measured attack arms against the layers.

What makes the difference between a defence tool and a bag of heuristics
is a harness that reports the arms where detection is poor with the same
prominence as the arms where it is good. Two rules:

* **retrieval failure and outcome failure stay separate**, because every
  ported layer is a vote-time filter and none of them changed retrieval at
  all in the research grid;
* **detection is reported at a stated budget**, with the genuine-record
  cost beside it. A detection rate without its false-positive cost is not
  a number anyone can act on.

The arms implemented here are the cell-corpus ones, replayed against a
corpus the caller supplies. The protein arms need the ProDVa attack inputs
from the research repository and are listed as not yet wired rather than
quietly omitted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core.arrays import Array
from .core.record import Record, Verdict
from .demo import DemoCorpus, poison
from .guard import Guard

#: The insert-arm attacks, by the research's names.
INSERT_ARMS = ("relabel", "duplicate", "interpolate", "flood", "hub")
#: Arms that need inputs this package does not ship.
NOT_WIRED = {
    "unmoved_inplace": "overwrite arm; use `bioguard manifest verify` -- no vote-time "
                       "layer detects it, which is the measured result",
    "shifted_inplace": "overwrite arm; needs an edited corpus rather than a batch",
    "synthesised_inplace": "overwrite arm; needs an edited corpus rather than a batch",
    "prodva_paraphrase": "needs the ProDVa attack inputs from the research repository",
    "prodva_nearduplicate": "needs the ProDVa attack inputs from the research repository",
    "prodva_hubs": "needs the ProDVa attack inputs from the research repository",
}


@dataclass
class ArmResult:
    """One attack arm against one configuration."""

    arm: str
    n_poison: int
    detected: int
    layers: dict[str, int] = field(default_factory=dict)
    budget: float = 0.05
    genuine_flagged: float = 0.0

    @property
    def detection_rate(self) -> float:
        return self.detected / self.n_poison if self.n_poison else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "n_poison": self.n_poison,
            "detected": self.detected,
            "detection_rate": self.detection_rate,
            "budget": self.budget,
            "genuine_records_flagged": self.genuine_flagged,
            "per_layer": self.layers,
        }


def _records(vectors: Array, rows: list[dict[str, Any]]) -> list[Record]:
    from datetime import datetime

    out = []
    for i, r in enumerate(rows):
        added = r.get("added_at")
        counts = r.get("counts")
        out.append(Record(
            id=r["id"], vector=vectors[i], label=r.get("label"), donor=r.get("donor"),
            counts=None if counts is None else np.asarray(counts, dtype=np.float32),
            added_at=None if added is None else datetime.fromisoformat(added),
            source=r.get("source"),
        ))
    return out


def evaluate_insert_arms(
    guard: Guard,
    corpus: DemoCorpus,
    arms: Sequence[str] = INSERT_ARMS,
    n: int = 30,
    budget: float = 0.05,
) -> list[ArmResult]:
    """Detection rate per arm at a fixed budget, with the cost beside it."""
    if guard.baseline is None:
        guard.calibrate(budget=budget)
    assert guard.baseline is not None
    genuine_flagged = float(np.mean([
        lb.flagged_fraction for lb in guard.baseline.layers.values()
    ])) if guard.baseline.layers else 0.0

    results = []
    for arm in arms:
        vectors, rows = poison(corpus, arm, n=n)
        report = guard.check_many(_records(vectors, rows))
        detected = sum(
            1 for r in report.results if r.verdict is not Verdict.ADMIT
        )
        results.append(ArmResult(
            arm=arm, n_poison=len(rows), detected=detected,
            layers=report.by_layer(), budget=budget,
            genuine_flagged=genuine_flagged,
        ))
    return results


def render_table(results: Sequence[ArmResult]) -> str:
    """A markdown table, with the not-wired arms named rather than dropped."""
    lines = [
        "| arm | n | detected at ingest | detection rate | layers that fired |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        fired = ", ".join(f"`{k}` x{v}" for k, v in r.layers.items()) or "none"
        lines.append(
            f"| `{r.arm}` | {r.n_poison} | {r.detected} | {r.detection_rate:.2f} | {fired} |"
        )
    lines.append("")
    lines.append("Arms not covered by this harness:")
    lines.append("")
    for arm, why in NOT_WIRED.items():
        lines.append(f"* `{arm}` -- {why}")
    return "\n".join(lines)
