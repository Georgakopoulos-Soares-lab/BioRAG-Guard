"""The ported neighbourhood scorer must agree with the research code.

``score_records`` carries two decisions that quietly change how strong the
defence looks -- self-matches excluded, and an empty neighbourhood scoring
1.0 rather than 0.0. Both are asserted here against the vendored original,
including the ``duplicate`` case that motivates the first.
"""

from __future__ import annotations

import numpy as np
import pytest
import upstream_label_consistency as up

from biorag_guard.layers.scoring import score_records

LABELS = np.array(["T cell", "B cell", "monocyte", "fibroblast"])


def _corpus(rng, n_ref=400, d=16):
    v = rng.normal(size=(n_ref, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float32), rng.choice(LABELS, size=n_ref)


@pytest.mark.parametrize("k", [1, 5, 50])
@pytest.mark.parametrize("block_size", [7, 500])
def test_matches_upstream(k, block_size, rng):
    ref_emb, ref_lab = _corpus(rng)
    q = rng.normal(size=(60, 16)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q_lab = rng.choice(LABELS, size=60)
    mine = score_records(q, q_lab, ref_emb, ref_lab, k=k, block_size=block_size)
    theirs = up.score_records(q, q_lab, ref_emb, ref_lab, k=k, block_size=block_size)
    np.testing.assert_allclose(mine, theirs, rtol=0, atol=0)


def test_duplicate_poison_does_not_vouch_for_itself(rng):
    """A verbatim copy of a reference row must be scored by its NEIGHBOURS."""
    ref_emb, ref_lab = _corpus(rng)
    # take a genuine record and resubmit it under another label
    victim = 11
    poison = ref_emb[victim:victim + 1].copy()
    forged = np.array(["fibroblast" if ref_lab[victim] != "fibroblast" else "T cell"])
    mine = score_records(poison, forged, ref_emb, ref_lab, k=50)
    theirs = up.score_records(poison, forged, ref_emb, ref_lab, k=50)
    np.testing.assert_allclose(mine, theirs, rtol=0, atol=0)
    # the self-match is excluded, so the forged label cannot score 1.0 off itself
    assert mine[0] < 1.0


def test_genuine_record_scores_higher_than_relabelled_one(rng):
    """The signal the defence acts on, stated as a test."""
    d = 8
    centres = {"T cell": np.eye(d)[0], "B cell": np.eye(d)[1]}
    ref, lab = [], []
    for name, c in centres.items():
        pts = c + 0.05 * rng.normal(size=(200, d))
        ref.append(pts)
        lab += [name] * 200
    ref_emb = np.vstack(ref).astype(np.float32)
    ref_emb /= np.linalg.norm(ref_emb, axis=1, keepdims=True)
    ref_lab = np.array(lab)

    probe = ref_emb[5:6]                       # sits among T cells
    honest = score_records(probe, np.array(["T cell"]), ref_emb, ref_lab, k=50)
    relabelled = score_records(probe, np.array(["B cell"]), ref_emb, ref_lab, k=50)
    assert honest[0] > 0.9
    assert relabelled[0] < 0.1


def test_empty_reference_scores_one():
    out = score_records(
        np.zeros((3, 4), dtype=np.float32), np.array(["a", "b", "c"]),
        np.zeros((0, 4), dtype=np.float32), np.array([], dtype=object),
    )
    np.testing.assert_array_equal(out, np.ones(3))


def test_empty_query_returns_empty():
    ref = np.eye(4, dtype=np.float32)
    out = score_records(np.zeros((0, 4), dtype=np.float32), np.array([]), ref,
                        np.array(["a", "b", "c", "d"]))
    assert out.shape == (0,)
