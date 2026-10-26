"""Neighbourhood scoring shared by the ported layers.

``score_records`` is ported from ``00_common/label_consistency.py`` (MIT).
Two decisions in it change how strong the defence looks, so both are kept
verbatim and both are tested.

**A record never scores itself.** Every record is its own nearest
neighbour and always agrees with itself. Counting that inflates every
score, and a lone poisoned record would partly vouch for itself. It matters
most for ``duplicate`` poison, which IS a verbatim copy of a reference row.

**Poison is scored against the CLEAN reference.** A defender checks a
submission against the database as it stands before accepting it. Scoring
against a reference that already holds the batch lets twenty copies of one
relabelled cell corroborate each other, which would make the defence look
useless as an artefact of the evaluation. The assumption this encodes is
that submissions are screened in batches rather than after bulk ingestion.
"""

from __future__ import annotations

import numpy as np

from ..core.arrays import Array

EMB_DTYPE = np.float32


def score_records(
    query_emb: Array,
    query_labels: Array,
    ref_emb: Array,
    ref_labels: Array,
    k: int = 50,
    block_size: int = 500,
) -> Array:
    """Fraction of each record's nearest reference neighbours sharing its label.

    Args:
        query_emb: (n, d) float32 unit norm -- the records being scored.
        query_labels: (n,) str -- the label each record CLAIMS.
        ref_emb: (n_ref, d) float32 unit norm -- the CLEAN reference.
        ref_labels: (n_ref,) str.
        k: neighbours to consult.
        block_size: records per block; arithmetically inert.

    Returns:
        (n,) float64 in [0, 1]. Higher means the record's own neighbourhood
        agrees with the label it carries.
    """
    query_emb = np.asarray(query_emb)
    ref_emb = np.asarray(ref_emb)
    query_labels = np.asarray(query_labels)
    ref_labels = np.asarray(ref_labels)
    n = len(query_emb)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    if len(ref_emb) == 0:
        out[:] = 1.0
        return out
    take = min(k + 1, len(ref_emb))  # +1 to absorb a self-match

    for lo in range(0, n, block_size):
        hi = min(lo + block_size, n)
        cos = (query_emb[lo:hi] @ ref_emb.T).astype(EMB_DTYPE)
        top = np.argpartition(-cos, take - 1, axis=1)[:, :take]
        tc = np.take_along_axis(cos, top, axis=1)
        order = np.argsort(-tc, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        tc = np.take_along_axis(tc, order, axis=1)

        for r in range(hi - lo):
            ids, sims = top[r], tc[r]
            # drop an exact self-match, then keep k
            keep = ids[~np.isclose(sims, 1.0, atol=1e-6)][:k]
            if keep.size == 0:
                keep = ids[1:k + 1]        # everything tied at 1.0; skip rank 0
            if keep.size == 0:
                out[lo + r] = 1.0          # no evidence is not evidence against
                continue
            out[lo + r] = float((ref_labels[keep] == query_labels[lo + r]).mean())
    return out


def max_similarity(
    query_emb: Array,
    ref_emb: Array,
    exclude_self: bool = True,
    block_size: int = 500,
) -> tuple[Array, Array]:
    """Highest cosine from each query to the reference, and which record it was.

    Args:
        query_emb: (n, d) unit-norm.
        ref_emb: (n_ref, d) unit-norm.
        exclude_self: drop an exact self-match (cos ~ 1.0 at the top rank),
            so a record already in the reference does not report itself as
            its own duplicate. Set False when screening an arriving batch
            against a reference that genuinely does not contain it.

    Returns:
        ``(best_cos, best_idx)``, each (n,). ``best_idx`` is -1 where the
        reference had nothing left to compare against.
    """
    query_emb = np.asarray(query_emb)
    ref_emb = np.asarray(ref_emb)
    n = len(query_emb)
    best = np.full(n, -1.0, dtype=np.float64)
    who = np.full(n, -1, dtype=np.int64)
    if n == 0 or len(ref_emb) == 0:
        return best, who

    for lo in range(0, n, block_size):
        hi = min(lo + block_size, n)
        cos = (query_emb[lo:hi] @ ref_emb.T).astype(EMB_DTYPE)
        if exclude_self:
            cos = np.where(np.isclose(cos, 1.0, atol=1e-6), -np.inf, cos)
        idx = np.argmax(cos, axis=1)
        val = np.take_along_axis(cos, idx[:, None], axis=1).ravel()
        finite = np.isfinite(val)
        best[lo:hi] = np.where(finite, val, -1.0)
        who[lo:hi] = np.where(finite, idx, -1)
    return best, who
