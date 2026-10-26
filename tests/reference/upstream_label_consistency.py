"""Score each record against its own neighbourhood — the defence's input.

# No `from __future__ import annotations` here: snakemake's script directive
# prepends a preamble, which would push it off line 1 and make it a SyntaxError.

A record's score is the fraction of its own nearest reference neighbours that
carry the same label. A relabelled cell sits among its true type while claiming
another, so it scores near zero; a genuine record sits among its own kind and
scores high. This is the signal the ``label_consistency`` defence acts on, and
the reason that defence works where embedding-space outlier detection does not:
the poison's VECTOR is genuine, but the pairing of that vector with that label
is not, and the neighbourhood exposes the pairing.

Two decisions here change how strong the defence appears, so both are explicit.

**A record never scores itself.** Every record is its own nearest neighbour and
always agrees with itself. Counting that inflates every score, and a lone
poisoned record would partly vouch for itself.

**Poison is scored against the CLEAN reference**, not against the reference with
the poison already in it. A defender checks a submission against the database as
it stands before accepting it. Scoring against the poisoned reference would let
twenty copies of one relabelled cell corroborate each other and make the defence
look useless — which would be an artefact of the evaluation, not a property of
the defence. The assumption this encodes is that submissions are screened in
batches rather than after bulk ingestion; a defender who ingests first and
checks later gets the weaker version.
"""

import numpy as np
from schemas import EMB_DTYPE


def score_records(
    query_emb: np.ndarray,
    query_labels: np.ndarray,
    ref_emb: np.ndarray,
    ref_labels: np.ndarray,
    k: int = 50,
    block_size: int = 500,
) -> np.ndarray:
    """Fraction of each record's nearest reference neighbours sharing its label.

    Args:
        query_emb: (n, d) float32 unit norm — the records being scored. For the
            defence these are poisoned records and the genuine reference records
            that appear in query neighbourhoods.
        query_labels: (n,) str — the label each record CLAIMS.
        ref_emb: (n_ref, d) float32 unit norm — the clean reference to score
            against. See the module note on why it must be clean.
        ref_labels: (n_ref,) str.
        k: neighbours to consult.
        block_size: records per block; arithmetically inert.

    Returns:
        (n,) float64 in [0, 1]. Higher means the record's own neighbourhood
        agrees with the label it carries.

    Note:
        A record identical to a reference row would otherwise match itself at
        cosine 1.0. Exact self-matches are dropped before scoring, so a record
        cannot vouch for itself — which matters because ``duplicate`` poison IS
        a verbatim copy of a reference row.
    """
    n = len(query_emb)
    out = np.empty(n, dtype=np.float64)
    take = min(k + 1, len(ref_emb))          # +1 to absorb a self-match

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
                keep = ids[1:k + 1]          # everything tied at 1.0; skip rank 0
            if keep.size == 0:
                out[lo + r] = 1.0            # no evidence is not evidence against
                continue
            out[lo + r] = float((ref_labels[keep] == query_labels[lo + r]).mean())
    return out
