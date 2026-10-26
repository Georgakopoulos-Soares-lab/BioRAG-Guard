"""Defence layers: filter the retrieved neighbourhood, then vote.

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

Ported from the protein arm of BioRAG-Guard (Phase 9), adapted to single-cell.
Each layer targets a MEASURED attack rather than a plausible-sounding threat,
and each is separately switchable so its marginal contribution is measurable
rather than only the full stack's:

``recompute``
    Admit a record only if its embedding is reproducible from real underlying
    counts. This kills the ``interpolate`` ceiling outright and does nothing
    whatever against ``relabel`` or ``duplicate``, whose vectors are verbatim
    reference embeddings. That asymmetry is the point: the gap between the
    ceiling and the realistic tiers measures what re-derivation buys.

``label_consistency``
    Flag a record whose attached label disagrees with the majority label of its
    OWN neighbourhood. A relabelled cell sits among its true type while carrying
    another type's label, so its agreement score is near zero. This is the layer
    that should defeat the realistic attack.

``donor_vote``
    One vote per donor rather than per cell. ``duplicate`` works by
    concentrating copies; collapsing them to a single vote removes the
    mechanism. It also corrects a pre-existing bias, since the shipped
    reference already contains duplicate embeddings.

``dedup``
    Collapse near-duplicate records inside a retrieved neighbourhood to a single
    vote, keeping the highest-ranked member of each group. This is the geometric
    analogue of ``donor_vote``, and its one structural advantage over every other
    layer here is that it reads **no metadata**: there is no donor id, no
    submitter, no timestamp for an attacker to fabricate. It acts on the vectors,
    which are the one thing the attacker must actually commit to.

``weight``
    Weight each surviving neighbour by its proximity instead of giving all k an
    equal vote. This is the one layer here that is not a filter: nothing is
    rejected, the arithmetic of the vote changes. Whether it helps is not
    obvious in advance — the cluster-flood places its records *closer* to the
    query than genuine neighbours are, so proximity weighting may well amplify
    exactly the attack it is proposed against. Measure it, do not assume it.

``abstain``
    Decline on a thin margin rather than guess. Declining is a legitimate
    outcome for an annotation system and is not counted as an error.

Nothing here retrains or learns. Every decision is a rule a human can be told
in one sentence, and every rejection is reconstructable from the neighbour
trace.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DefenseConfig:
    """Which layers are active. Composable so each can be measured alone."""

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


#: Floor on ``1 - cos`` in the SCimilarity score. A poisoned record copied
#: verbatim can land at cos = 1.0 in float32, and the score 1 / (1 - c) is then
#: infinite; one such neighbour would silently decide every vote. Clipping is an
#: explicit modelling choice, stated here, not a silent numerical fallback: the
#: cap is 1e6, far above any weight a genuine neighbour attains.
SCORE_EPS = 1e-6

WEIGHT_SCHEMES = ("uniform", "score", "cosine", "rank", "softmax")


def vote_weights(cos: np.ndarray, cfg: DefenseConfig) -> np.ndarray:
    """Per-neighbour vote weights under the configured scheme.

    Args:
        cos: (m,) cosine similarities of the surviving neighbours, in rank
            order — descending, so index 0 is the nearest.
        cfg: which scheme, and the softmax temperature.

    Returns:
        (m,) float64 non-negative weights. ``uniform`` returns all ones, which
        reproduces the majority vote exactly.

    Raises:
        ValueError: unknown scheme name. Falling back to uniform would report a
            weighting that never ran as a weighting that made no difference.
    """
    c = np.asarray(cos, dtype=np.float64)
    if cfg.weight == "uniform":
        return np.ones(len(c))
    if cfg.weight == "score":
        # the SCimilarity score itself, 1 / (1 - c); see CLAUDE.md §1
        return 1.0 / np.maximum(1.0 - c, SCORE_EPS)
    if cfg.weight == "cosine":
        # shifted so an antipodal neighbour contributes nothing rather than
        # a negative vote, which is not meaningful in a tally
        return np.maximum(c, 0.0)
    if cfg.weight == "rank":
        return 1.0 / (1.0 + np.arange(len(c)))
    if cfg.weight == "softmax":
        z = (c - c.max()) / cfg.weight_temp
        return np.exp(z)
    raise ValueError(f"unknown weight scheme {cfg.weight!r}; expected one of {WEIGHT_SCHEMES}")


def dedup_keep(emb: np.ndarray, tau: float) -> np.ndarray:
    """Indices surviving greedy near-duplicate collapse, in rank order.

    Walks the neighbourhood from nearest to furthest and keeps a record only if
    it is below ``tau`` cosine to everything kept so far. The survivor of a
    duplicate group is therefore its highest-ranked member.

    **This is only a defence at ingestion. See HANDOFF 3.25.** An exact copy sits
    at the SAME cosine to the query as its original, so "highest-ranked" is a tie
    and the tie falls to whatever order the caller built the array in. Listing
    the poison first makes this layer drop the genuine record and keep the
    poisoned one, and the defended result becomes identical to the undefended
    one. At ingestion the asymmetry is real — the index is trusted, the arriving
    batch is not — and the rule is principled. At query time both records are
    already in the index and nothing here can tell them apart.

    Args:
        emb: (m, d) float32 unit vectors of the neighbours, ranked descending by
            similarity to the query.
        tau: cosine at or above which two records count as the same record.

    Returns:
        (m',) int indices into ``emb``, ascending, with m' <= m. Always keeps at
        least the first record.
    """
    e = np.asarray(emb, dtype=np.float32)
    # One gram matrix, then a greedy walk over it. The original form recomputed
    # ``e[keep] @ e[i]`` inside the loop, which is the same arithmetic in 50
    # tiny matmuls per neighbourhood -- 2.5M numpy calls across a full query
    # set, where call overhead rather than flops dominated. Rewritten
    # 2026-09-29 for 03b_01; ``test_defense`` pins the two forms equal.
    g = e @ e.T
    keep: list = []
    for i in range(len(e)):
        if not keep or float(g[i, keep].max()) < tau:
            keep.append(i)
    return np.array(keep, dtype=int)


def label_agreement(attached: str, neighbor_labels) -> float:
    """Fraction of a record's own neighbours carrying the same label.

    Args:
        attached: the label the record claims.
        neighbor_labels: labels of that record's nearest neighbours in the
            reference, excluding itself.

    Returns:
        Agreement in [0, 1]. An empty neighbourhood returns 1.0: no evidence is
        not evidence against, and flagging on emptiness would reject records
        merely for being isolated.
    """
    labels = list(neighbor_labels)
    if not labels:
        return 1.0
    return sum(1 for x in labels if x == attached) / len(labels)


def admit(
    cfg: DefenseConfig,
    recompute_ok: bool = True,
    label_consistency: float = 1.0,
    seasoned: bool = True,
) -> bool:
    """Whether one neighbour survives the record-level filters.

    Args:
        cfg: which layers are active.
        recompute_ok: is this record's embedding reproducible from real counts?
        label_consistency: agreement between its label and its neighbourhood.
        seasoned: has this record been in the reference longer than the
            quarantine window? Recently added records do not vote.

    Returns:
        True if every enabled filter passes.
    """
    if cfg.recompute and not recompute_ok:
        return False
    if cfg.label_consistency and label_consistency < cfg.lc_threshold:
        return False
    if cfg.quarantine and not seasoned:
        return False
    return True


def survivors(
    labels: np.ndarray,
    cfg: DefenseConfig,
    recompute_ok: np.ndarray | None = None,
    label_consistency: np.ndarray | None = None,
    donor: np.ndarray | None = None,
    seasoned: np.ndarray | None = None,
    emb: np.ndarray | None = None,
) -> np.ndarray:
    """Indices of one query's neighbours that survive every enabled layer.

    Split out of :func:`defended_vote` so a caller can ask what a defence
    REMOVED, not only what it voted. Reporting a defended retrieval rate off the
    raw neighbourhood would credit every filter with zero effect on retrieval.

    Args:
        labels: (k,) labels of the merged top-k, ranked. Other arguments as for
        :func:`defended_vote`.

    Returns:
        Ascending indices into ``labels``, possibly empty.
    """
    n = len(np.asarray(labels))
    ok = np.ones(n, dtype=bool)
    for i in range(n):
        ok[i] = admit(
            cfg,
            recompute_ok=bool(recompute_ok[i]) if recompute_ok is not None else True,
            label_consistency=(
                float(label_consistency[i]) if label_consistency is not None else 1.0
            ),
            seasoned=bool(seasoned[i]) if seasoned is not None else True,
        )
    kept = np.flatnonzero(ok)
    if kept.size == 0:
        return kept

    if cfg.dedup_tau:
        kept = kept[dedup_keep(np.asarray(emb)[kept], cfg.dedup_tau)]

    if cfg.donor_vote or cfg.donor_cap:
        cap = cfg.donor_cap if cfg.donor_cap else 1
        seen: dict = {}
        keep = []
        for i in kept:
            d = str(donor[i])
            if seen.get(d, 0) < cap:
                seen[d] = seen.get(d, 0) + 1
                keep.append(i)
        kept = np.array(keep, dtype=int)
    return kept


def defended_vote(
    labels: np.ndarray,
    cfg: DefenseConfig,
    recompute_ok: np.ndarray | None = None,
    label_consistency: np.ndarray | None = None,
    donor: np.ndarray | None = None,
    seasoned: np.ndarray | None = None,
    cos: np.ndarray | None = None,
    emb: np.ndarray | None = None,
) -> str | None:
    """Filter one query's neighbourhood, then vote over what survives.

    Args:
        labels: (k,) labels of the merged top-k, ranked.
        cfg: which layers are active.
        recompute_ok: (k,) bool, required when ``cfg.recompute``.
        label_consistency: (k,) float, required when ``cfg.label_consistency``.
        donor: (k,) str, required when ``cfg.donor_vote`` or ``cfg.donor_cap``.
        seasoned: (k,) bool, required when ``cfg.quarantine`` — False for a
            record added within the quarantine window.
        cos: (k,) float cosine similarities, ranked descending, required when
            ``cfg.weight`` is not ``uniform``.
        emb: (k, d) float32 unit vectors of the neighbours, required when
            ``cfg.dedup_tau`` is set.

    Returns:
        The predicted label, or ``None`` when the defence abstains or filtering
        leaves nothing. ``None`` is a refusal to answer, not a wrong answer, and
        must be scored as such.

    Raises:
        ValueError: a layer is enabled but its evidence is absent. Treating a
            missing score as passing would report a defence that never ran as a
            defence that found nothing.
    """
    labels = np.asarray(labels)
    if cfg.recompute and recompute_ok is None:
        raise ValueError("recompute is enabled but recompute_ok was not supplied")
    if cfg.label_consistency and label_consistency is None:
        raise ValueError("label_consistency is enabled but no scores were supplied")
    if (cfg.donor_vote or cfg.donor_cap) and donor is None:
        raise ValueError("donor voting is enabled but no donor ids were supplied")
    if cfg.quarantine and seasoned is None:
        raise ValueError("quarantine is enabled but no record ages were supplied")
    if cfg.dedup_tau and emb is None:
        raise ValueError("dedup_tau is set but no neighbour embeddings were supplied")

    kept = survivors(labels, cfg, recompute_ok=recompute_ok,
                     label_consistency=label_consistency, donor=donor,
                     seasoned=seasoned, emb=emb)
    if kept.size == 0:
        return None

    if cfg.weight != "uniform" and cos is None:
        raise ValueError(f"weight={cfg.weight!r} is enabled but no cosines were supplied")
    w = vote_weights(np.asarray(cos)[kept] if cos is not None else np.zeros(kept.size), cfg)
    tally: dict = {}
    for lab, wi in zip(labels[kept].tolist(), w.tolist()):
        tally[lab] = tally.get(lab, 0.0) + wi
    ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(tally.values())
    if total <= 0.0:
        return None
    if cfg.abstain:
        top = ranked[0][1] / total
        second = ranked[1][1] / total if len(ranked) > 1 else 0.0
        if top - second < cfg.abstain_min_margin:
            return None
    return ranked[0][0]
