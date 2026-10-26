"""Filter a retrieved neighbourhood, then vote over what survives.

Ported from ``00_common/defense.py`` (MIT). Field names, defaults and the
``name`` string are preserved so a configuration is directly comparable with
the research tables.

Two semantics are easy to lose in a port and are load-bearing:

**dedup is sound at ingest only.** An exact copy sits at the SAME cosine to
the query as its original, so "keep the highest-ranked member" is a tie, and
the tie falls to whatever order the caller built the array in. Listing the
poison first makes the layer drop the genuine record and keep the poisoned
one. At ingest the asymmetry is real -- the index is trusted, the arriving
batch is not -- and the rule is principled. At query time both records are
already in the index and nothing here can tell them apart.

**A missing score is not a pass.** Every enabled layer whose evidence is
absent raises. Reporting a defence that never ran as a defence that found
nothing is the one failure mode that cannot be recovered downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import MissingEvidence
from .weights import vote_weights


@dataclass(frozen=True)
class VotePolicy:
    """Which layers are active at vote time. Composable, so each is measurable alone.

    The field names are those of the research ``DefenseConfig``.
    """

    recompute: bool = False
    label_consistency: bool = False
    lc_threshold: float = 0.5
    donor_vote: bool = False
    donor_cap: int = 0
    quarantine: bool = False
    dedup_tau: float = 0.0
    weight: str = "uniform"
    weight_temp: float = 0.01
    abstain: bool = False
    abstain_min_margin: float = 0.0

    @property
    def name(self) -> str:
        """Short label for reporting, e.g. ``recompute+donor``."""
        parts = []
        if self.recompute:
            parts.append("recompute")
        if self.label_consistency:
            parts.append("labelcons")
        if self.donor_vote:
            parts.append("donor")
        if self.donor_cap:
            parts.append(f"donorcap{self.donor_cap}")
        if self.quarantine:
            parts.append("quarantine")
        if self.dedup_tau:
            parts.append(f"dedup{self.dedup_tau:g}")
        if self.weight != "uniform":
            parts.append(f"w:{self.weight}")
        if self.abstain:
            parts.append("abstain")
        return "+".join(parts) if parts else "none"


def dedup_keep(emb: np.ndarray, tau: float) -> np.ndarray:
    """Indices surviving greedy near-duplicate collapse, in rank order.

    Walks the neighbourhood from nearest to furthest and keeps a record only
    if it is below ``tau`` cosine to everything kept so far, so the survivor
    of a duplicate group is its highest-ranked member.

    Args:
        emb: (m, d) float32 unit vectors, ranked descending by similarity to
            the query.
        tau: cosine at or above which two records count as the same record.

    Returns:
        (m',) ascending indices into ``emb``, m' <= m. Always keeps at least
        the first record.
    """
    e = np.asarray(emb, dtype=np.float32)
    if len(e) == 0:
        return np.array([], dtype=int)
    gram = e @ e.T
    keep: list[int] = []
    for i in range(len(e)):
        if not keep or float(gram[i, keep].max()) < tau:
            keep.append(i)
    return np.array(keep, dtype=int)


def label_agreement(attached: str, neighbor_labels: object) -> float:
    """Fraction of a record's own neighbours carrying the same label.

    Args:
        attached: the label the record claims.
        neighbor_labels: labels of that record's nearest neighbours in the
            reference, excluding itself.

    Returns:
        Agreement in [0, 1]. An empty neighbourhood returns 1.0: no evidence
        is not evidence against, and flagging on emptiness would reject
        records merely for being isolated.
    """
    labels = list(neighbor_labels)  # type: ignore[call-overload]
    if not labels:
        return 1.0
    return sum(1 for x in labels if x == attached) / len(labels)


def admit(
    policy: VotePolicy,
    recompute_ok: bool = True,
    label_consistency: float = 1.0,
    seasoned: bool = True,
) -> bool:
    """Whether one neighbour survives the record-level filters.

    Args:
        policy: which layers are active.
        recompute_ok: is this record's embedding reproducible from real counts?
        label_consistency: agreement between its label and its neighbourhood.
        seasoned: has the record been in the reference longer than the
            quarantine window? Recently added records do not vote.
    """
    if policy.recompute and not recompute_ok:
        return False
    if policy.label_consistency and label_consistency < policy.lc_threshold:
        return False
    if policy.quarantine and not seasoned:
        return False
    return True


def survivors(
    labels: np.ndarray,
    policy: VotePolicy,
    recompute_ok: np.ndarray | None = None,
    label_consistency: np.ndarray | None = None,
    donor: np.ndarray | None = None,
    seasoned: np.ndarray | None = None,
    emb: np.ndarray | None = None,
) -> np.ndarray:
    """Indices of one query's neighbours that survive every enabled layer.

    Separate from :func:`defended_vote` so a caller can ask what a defence
    REMOVED, not only what it voted. Reporting a defended retrieval rate off
    the raw neighbourhood would credit every filter with zero effect on
    retrieval.

    Returns:
        Ascending indices into ``labels``, possibly empty.
    """
    n = len(np.asarray(labels))
    ok = np.ones(n, dtype=bool)
    for i in range(n):
        ok[i] = admit(
            policy,
            recompute_ok=bool(recompute_ok[i]) if recompute_ok is not None else True,
            label_consistency=(
                float(label_consistency[i]) if label_consistency is not None else 1.0
            ),
            seasoned=bool(seasoned[i]) if seasoned is not None else True,
        )
    kept = np.flatnonzero(ok)
    if kept.size == 0:
        return kept

    if policy.dedup_tau:
        kept = kept[dedup_keep(np.asarray(emb)[kept], policy.dedup_tau)]

    if policy.donor_vote or policy.donor_cap:
        cap = policy.donor_cap if policy.donor_cap else 1
        seen: dict[str, int] = {}
        keep: list[int] = []
        for i in kept.tolist():
            d = str(donor[i])  # type: ignore[index]
            if seen.get(d, 0) < cap:
                seen[d] = seen.get(d, 0) + 1
                keep.append(int(i))
        kept = np.array(keep, dtype=int)
    return kept


def defended_vote(
    labels: np.ndarray,
    policy: VotePolicy,
    recompute_ok: np.ndarray | None = None,
    label_consistency: np.ndarray | None = None,
    donor: np.ndarray | None = None,
    seasoned: np.ndarray | None = None,
    cos: np.ndarray | None = None,
    emb: np.ndarray | None = None,
) -> str | None:
    """Filter one query's neighbourhood, then vote over what survives.

    Returns:
        The predicted label, or ``None`` when the defence abstains or
        filtering leaves nothing. ``None`` is a refusal to answer, not a
        wrong answer, and must be scored as such.

    Raises:
        MissingEvidence: a layer is enabled but its evidence is absent.
    """
    labels = np.asarray(labels)
    if policy.recompute and recompute_ok is None:
        raise MissingEvidence("recompute is enabled but recompute_ok was not supplied")
    if policy.label_consistency and label_consistency is None:
        raise MissingEvidence("label_consistency is enabled but no scores were supplied")
    if (policy.donor_vote or policy.donor_cap) and donor is None:
        raise MissingEvidence("donor voting is enabled but no donor ids were supplied")
    if policy.quarantine and seasoned is None:
        raise MissingEvidence("quarantine is enabled but no record ages were supplied")
    if policy.dedup_tau and emb is None:
        raise MissingEvidence("dedup_tau is set but no neighbour embeddings were supplied")

    kept = survivors(
        labels, policy, recompute_ok=recompute_ok, label_consistency=label_consistency,
        donor=donor, seasoned=seasoned, emb=emb,
    )
    if kept.size == 0:
        return None

    if policy.weight != "uniform" and cos is None:
        raise MissingEvidence(
            f"weight={policy.weight!r} is enabled but no cosines were supplied"
        )
    w = vote_weights(
        np.asarray(cos)[kept] if cos is not None else np.zeros(kept.size),
        policy.weight, policy.weight_temp,
    )
    tally: dict[str, float] = {}
    for lab, wi in zip(labels[kept].tolist(), w.tolist()):
        tally[lab] = tally.get(lab, 0.0) + wi
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(tally.values())
    if total <= 0.0:
        return None
    if policy.abstain:
        top = ranked[0][1] / total
        second = ranked[1][1] / total if len(ranked) > 1 else 0.0
        if top - second < policy.abstain_min_margin:
            return None
    return ranked[0][0]
