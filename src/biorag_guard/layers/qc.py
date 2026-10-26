"""Quality-control layers: cheap, broad, and honest about their reach.

These catch fabricated or mis-prepared submissions, not clever poison.
Most of the measured attack taxonomy is built from genuine profiles and
passes QC by construction, so the layers are worth having and must not be
oversold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.arrays import Array
from ..core.finding import Finding
from ..core.record import Access, Record, Scope, Severity, Verdict
from ..core.registry import Context, Direction, Layer, LayerSpec, register_layer


def count_stats(counts: Any) -> dict[str, float]:
    """Library size, features detected, and how integral the counts are."""
    c = np.asarray(counts, dtype=np.float64).ravel()
    total = float(c.sum())
    detected = float((c > 0).sum())
    residual = float(np.abs(c - np.round(c)).max()) if c.size else 0.0
    return {
        "library_size": total,
        "features_detected": detected,
        "detected_fraction": detected / max(c.size, 1),
        "max_integer_residual": residual,
        "max_value": float(c.max()) if c.size else 0.0,
    }


@register_layer
class QcCounts(Layer):
    """Submitted measurements that do not look like real measurements.

    Scored as the largest robust z-score across the basic statistics, so
    one wildly atypical library size is enough to fire without needing a
    separate threshold per statistic.
    """

    spec = LayerSpec(
        name="qc_counts",
        scope=Scope.RECORD_LOCAL,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"counts"}),
        catches="Fabricated profiles: impossible library sizes, implausible detection "
                "rates, non-integer raw counts.",
        misses="Poison built from real profiles, which is most of the measured attack "
               "taxonomy. This is a hygiene check, not a defence.",
        statistic="largest robust z-score (median / MAD) across the count statistics",
        direction=Direction.HIGH_IS_SUSPICIOUS,
        default_verdict=Verdict.FLAG,
        severity=Severity.LOW,
        default_params={"stats": ("library_size", "detected_fraction")},
    )

    def score(self, records: Sequence[Record], ctx: Context) -> Array:
        ref = ctx.cache.get("qc_counts:reference")
        table: Array = np.array(
            [[count_stats(r.counts)[s] for s in self.params["stats"]] for r in records],
            dtype=np.float64,
        )
        if ref is None:
            ref = table
        med = np.median(ref, axis=0)
        mad = np.median(np.abs(ref - med), axis=0)
        mad = np.where(mad <= 0, 1.0, mad)
        z = np.abs(table - med) / (1.4826 * mad)
        return np.asarray(z.max(axis=1), dtype=np.float64)

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        stats = count_stats(record.counts)
        return (
            f"This record's measurements sit {score:.1f} robust standard deviations from "
            f"the corpus: library size {stats['library_size']:.0f}, "
            f"{stats['features_detected']:.0f} features detected."
        )

    def remedy(self, record: Record) -> str:
        return "Re-check the submission's QC before accepting; it may be mis-exported."


@register_layer
class QcAlign(Layer):
    """Records prepared differently from the corpus they are joining.

    Not an attack detector. A submission embedded under a different
    feature order or normalisation produces a vector that means something
    else, and every downstream layer would then be measuring the wrong
    thing. Rejecting it prevents a run of confident nonsense.
    """

    spec = LayerSpec(
        name="qc_align",
        scope=Scope.RECORD_LOCAL,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"vector"}),
        catches="Wrong embedding dimension, non-finite values, and vectors that are not "
                "unit-norm when the corpus is.",
        misses="Anything correctly prepared. This is a correctness gate, not a defence.",
        statistic="dimension match, finiteness, and norm against the corpus convention",
        budgeted=False,
        default_verdict=Verdict.REJECT,
        severity=Severity.HIGH,
        default_params={"norm_tolerance": 1e-3},
    )

    def run(self, records: Sequence[Record], ctx: Context) -> list[Finding]:
        self.check_evidence(records)
        adapter = ctx.adapter
        dims = None
        if adapter is not None and adapter.vectors is not None:
            dims = int(np.asarray(adapter.vectors).shape[1])
        tol = float(self.params["norm_tolerance"])
        out: list[Finding] = []
        for rec in records:
            v = np.asarray(rec.vector, dtype=np.float64).ravel()
            problems = []
            if dims is not None and v.size != dims:
                problems.append(
                    f"it has {v.size} dimensions where the corpus has {dims}"
                )
            if not np.isfinite(v).all():
                problems.append("it contains values that are not finite")
            norm = float(np.linalg.norm(v))
            if np.isfinite(norm) and abs(norm - 1.0) > tol and norm > 0:
                problems.append(f"its length is {norm:.4f} rather than 1")
            if not problems:
                continue
            out.append(
                Finding(
                    layer=self.name, record_id=rec.id, score=float(len(problems)),
                    threshold=0.0, severity=self.spec.severity, verdict=self.verdict,
                    reason="This record was not prepared the way the corpus was: "
                           + ", and ".join(problems) + ".",
                    remedy="Re-run the corpus's own alignment and normalisation steps on "
                           "the submission, then resubmit.",
                    evidence={"dims": int(v.size), "corpus_dims": dims, "norm": norm},
                )
            )
        return out
