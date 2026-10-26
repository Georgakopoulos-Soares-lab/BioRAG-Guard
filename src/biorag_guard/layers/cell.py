"""The ported cell-corpus defences.

Each layer targets a MEASURED attack rather than a plausible-sounding
threat, and each is separately switchable so its marginal contribution is
visible rather than only the stack's. The numbers in the docstrings are
mean within-celltype attack success from ``03a_01_no_access_layer_matrix``
(insert arm, 26 celltypes x 4 attacks x 2 doses) and
``03b_01_with_access_layer_matrix`` (overwrite arm).

The headline asymmetry: ``recompute`` is the only layer that stops a
synthesised cluster flood and it does nothing at all against a relabel;
``dedup`` and ``label_consistency`` are the mirror image. A default policy
that enables one family ships a hole.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.arrays import Array
from ..core.record import Access, Record, Scope, Severity, Verdict
from ..core.registry import Context, Direction, Layer, LayerSpec, register_layer
from .scoring import max_similarity, score_records


@register_layer
class Recompute(Layer):
    """Admit a record only if its embedding is reproducible from real counts.

    Measured: takes ``flood`` from 0.264 to 0.001 and ``interpolate`` to
    0.001 in the insert arm, and ``shifted_inplace`` from 0.306 to 0.006 in
    the overwrite arm. Does **nothing** against ``duplicate`` and
    ``relabel`` (0.343 to 0.343), whose vectors are verbatim genuine
    embeddings. That asymmetry is the point.
    """

    spec = LayerSpec(
        name="recompute",
        scope=Scope.RECORD_LOCAL,
        access=frozenset({Access.INSERT, Access.OVERWRITE}),
        needs=frozenset({"vector", "counts"}),
        catches="Embeddings that no real measurement could have produced: synthesised "
                "geometry, interpolated vectors, and vectors written straight into the index.",
        misses="Poison built from genuine embeddings. A relabelled or duplicated real cell "
               "re-derives perfectly, so this layer passes it.",
        statistic="cosine distance between the stored vector and the vector re-derived "
                  "from the record's own counts",
        ported=True,
        direction=Direction.HIGH_IS_SUSPICIOUS,
        default_verdict=Verdict.QUARANTINE,
        severity=Severity.HIGH,
        default_params={"tolerance": 1e-4},
    )

    def fires(self, scores: Array) -> Array:
        """Compare against the calibrated threshold or the numerical floor.

        On a corpus whose vectors re-derive cleanly, the budget quantile lands
        at exactly 0.0 and every record with a float-rounding residue would
        fire. A re-encode never reproduces bit for bit, so a disagreement
        below ``tolerance`` is arithmetic, not evidence.
        """
        floor = float(self.params["tolerance"])
        return np.asarray(scores) > max(self.threshold, floor)

    def score(self, records: Sequence[Record], ctx: Context) -> Array:
        if ctx.adapter is None:
            raise NotImplementedError("recompute needs an adapter that can re-derive vectors")
        redone = ctx.adapter.recompute(records)
        if redone is None:
            raise NotImplementedError(
                "this corpus cannot re-derive embeddings from counts"
            )
        stored = np.vstack([np.asarray(r.vector, dtype=np.float32) for r in records])
        redone = np.asarray(redone, dtype=np.float32)
        cos = np.sum(stored * redone, axis=1)
        return np.asarray(1.0 - cos, dtype=np.float64)

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        return (
            f"The embedding stored for this record is not what its own measurements "
            f"produce when re-encoded (cosine {1 - score:.4f}). Either the counts and the "
            f"vector are not from the same cell, or the vector was written directly."
        )

    def remedy(self, record: Record) -> str:
        return "Re-encode from the submitted counts and compare; reject if they disagree."


@register_layer
class LabelConsistency(Layer):
    """Flag a record whose label disagrees with its OWN neighbourhood.

    A relabelled cell sits among its true type while carrying another
    type's label, so its agreement score is near zero. Measured: takes
    ``relabel`` and ``duplicate`` from 0.343 to 0.002 and ``flood`` from
    0.264 to 0.004.

    Scored against the corpus as it stood BEFORE the batch. If the batch is
    already in the index, copies of one relabelled cell corroborate each
    other; the layer refuses rather than reporting a weakened score.
    """

    spec = LayerSpec(
        name="label_consistency",
        scope=Scope.CORPUS_RELATIVE,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"vector", "label"}),
        catches="The realistic insert attack: a genuine embedding submitted under a label "
                "its neighbours do not share.",
        misses="A label rewritten on a record already in the corpus -- there is no "
               "submission to screen. Measured at 0.149 against 0.162 undefended, which is "
               "no effect. Use the manifest for that case.",
        statistic="fraction of the record's k nearest reference neighbours carrying the "
                  "same label, self-matches excluded",
        ported=True,
        direction=Direction.LOW_IS_SUSPICIOUS,
        default_verdict=Verdict.QUARANTINE,
        severity=Severity.HIGH,
        default_params={"k": 50},
    )

    def score(self, records: Sequence[Record], ctx: Context) -> Array:
        adapter = ctx.adapter
        if adapter is None or adapter.vectors is None or adapter.labels is None:
            raise NotImplementedError(
                "label_consistency needs a reference with vectors and labels"
            )
        q = np.vstack([np.asarray(r.vector, dtype=np.float32) for r in records])
        q_lab = np.array([r.label for r in records], dtype=object)
        return score_records(
            q, q_lab, np.asarray(adapter.vectors), np.asarray(adapter.labels),
            k=int(self.params["k"]),
        )

    def evidence(self, record: Record, index: int, ctx: Context) -> dict[str, Any]:
        adapter = ctx.adapter
        if adapter is None or adapter.vectors is None or adapter.labels is None:
            return {}
        k = int(self.params["k"])
        idx, cos = adapter.neighbors(np.asarray(record.vector)[None, :], k + 1)
        labs = np.asarray(adapter.labels)[idx[0]]
        keep = ~np.isclose(cos[0], 1.0, atol=1e-6)
        labs, sims = labs[keep][:k], cos[0][keep][:k]
        tally: dict[str, int] = {}
        for lab in labs.tolist():
            tally[str(lab)] = tally.get(str(lab), 0) + 1
        return {
            "k": int(len(labs)),
            "claimed_label": record.label,
            "neighbour_labels": dict(sorted(tally.items(), key=lambda kv: -kv[1])),
            "nearest_cosine": float(sims[0]) if len(sims) else None,
        }

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        k = evidence.get("k", self.params["k"])
        agree = int(round(score * k))
        top = next(iter(evidence.get("neighbour_labels", {})), None)
        tail = f" Most of them are {top!r}." if top and top != record.label else ""
        return (
            f"Of this record's {k} nearest neighbours in the corpus, {agree} carry the "
            f"label it was submitted under ({record.label!r}).{tail}"
        )

    def remedy(self, record: Record) -> str:
        return ("Confirm the annotation with the submitter before accepting; the embedding "
                "looks genuine, the label is what does not fit.")


@register_layer
class Dedup(Layer):
    """Collapse records that are near-identical to something already present.

    Measured: takes ``duplicate`` and ``relabel`` from 0.343 to 0.001, and
    it is the only layer that reads **no metadata** -- no donor id, no
    submitter, no timestamp for an attacker to fabricate. It acts on the
    vectors, which are the one thing the attacker must commit to.

    Leaves ``flood`` at 0.145 (0.219 at the higher dose), because a flood
    is many distinct records rather than copies.

    **Ingest only.** At query time an exact copy sits at the same cosine as
    its original, "keep the highest-ranked member" is a tie, and the order
    the caller built the array in decides which survives. At ingest the
    asymmetry is real: the corpus is trusted, the arriving batch is not.
    """

    spec = LayerSpec(
        name="dedup",
        scope=Scope.CORPUS_RELATIVE,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"vector"}),
        catches="Verbatim and near-verbatim copies of records already in the corpus, which "
                "is how a budgeted attacker concentrates influence on one neighbourhood.",
        misses="A cluster flood of distinct synthesised records. Unsound at query time -- "
               "see the class docstring.",
        statistic="highest cosine from the arriving record to any record already present",
        ported=True,
        direction=Direction.HIGH_IS_SUSPICIOUS,
        default_verdict=Verdict.QUARANTINE,
        severity=Severity.HIGH,
        ingest_only=True,
        default_params={"tau": 0.999},
    )

    def score(self, records: Sequence[Record], ctx: Context) -> Array:
        adapter = ctx.adapter
        if adapter is None or adapter.vectors is None:
            raise NotImplementedError("dedup needs a reference with vectors")
        q = np.vstack([np.asarray(r.vector, dtype=np.float32) for r in records])
        in_corpus = set(np.asarray(adapter.ids, dtype=object).tolist())
        exclude_self = any(r.id in in_corpus for r in records)
        best, _ = max_similarity(q, np.asarray(adapter.vectors), exclude_self=exclude_self)
        return best

    def evidence(self, record: Record, index: int, ctx: Context) -> dict[str, Any]:
        adapter = ctx.adapter
        if adapter is None or adapter.vectors is None:
            return {}
        in_corpus = str(record.id) in set(np.asarray(adapter.ids, dtype=object).tolist())
        best, who = max_similarity(
            np.asarray(record.vector)[None, :], np.asarray(adapter.vectors),
            exclude_self=in_corpus,
        )
        j = int(who[0])
        return {
            "nearest_existing_id": None if j < 0 else str(np.asarray(adapter.ids)[j]),
            "cosine": float(best[0]),
        }

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        other = evidence.get("nearest_existing_id")
        where = f" ({other})" if other else ""
        return (
            f"This record is a near-identical copy of one already in the corpus{where}, "
            f"at cosine {score:.6f}. Accepting it gives one measurement two votes."
        )

    def remedy(self, record: Record) -> str:
        return "Keep the record already present and drop the copy, or merge their metadata."
