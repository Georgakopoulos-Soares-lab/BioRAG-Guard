"""The ported vote must agree with the research implementation, exactly.

Ported layers are only worth the name if they behave identically. These
tests run the vendored ``00_common/defense.py`` beside
``biorag_guard.vote`` over randomised neighbourhoods and assert equality --
including on the awkward cases that a reimplementation typically gets
wrong: the float32 cos=1.0 clip, an empty survivor set, the tie-break, and
raising rather than passing when evidence is missing.
"""

from __future__ import annotations

import numpy as np
import pytest
import upstream_defense as up

from biorag_guard.core.errors import MissingEvidence
from biorag_guard.vote import (
    SCORE_EPS,
    WEIGHT_SCHEMES,
    VotePolicy,
    admit,
    dedup_keep,
    defended_vote,
    label_agreement,
    survivors,
    vote_weights,
)

LABELS = np.array(["A", "B", "C", "D"])


def _policies():
    """A grid of configurations covering every layer alone and in composition."""
    out = [VotePolicy()]
    out += [
        VotePolicy(recompute=True),
        VotePolicy(label_consistency=True, lc_threshold=0.14),
        VotePolicy(label_consistency=True, lc_threshold=0.5),
        VotePolicy(donor_vote=True),
        VotePolicy(donor_cap=1),
        VotePolicy(donor_cap=2),
        VotePolicy(donor_cap=5),
        VotePolicy(quarantine=True),
        VotePolicy(dedup_tau=0.999),
        VotePolicy(dedup_tau=0.9),
        VotePolicy(abstain=True, abstain_min_margin=0.1),
    ]
    for scheme in WEIGHT_SCHEMES:
        out.append(VotePolicy(weight=scheme))
    # the full stack
    out.append(
        VotePolicy(recompute=True, label_consistency=True, lc_threshold=0.14,
                   donor_cap=1, dedup_tau=0.999, weight="score",
                   abstain=True, abstain_min_margin=0.05)
    )
    return out


def _as_upstream(p: VotePolicy) -> up.DefenseConfig:
    return up.DefenseConfig(
        recompute=p.recompute, label_consistency=p.label_consistency,
        lc_threshold=p.lc_threshold, donor_vote=p.donor_vote, donor_cap=p.donor_cap,
        quarantine=p.quarantine, dedup_tau=p.dedup_tau, weight=p.weight,
        weight_temp=p.weight_temp, abstain=p.abstain,
        abstain_min_margin=p.abstain_min_margin,
    )


def _neighbourhood(rng, k=50, d=16):
    v = rng.normal(size=(k, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # force duplicates so dedup has something to collapse
    v[1] = v[0]
    v[7] = v[6]
    cos = np.sort(rng.uniform(0.6, 1.0, size=k))[::-1].astype(np.float64)
    cos[0] = 1.0  # exercises the SCORE_EPS clip
    return {
        "labels": rng.choice(LABELS, size=k),
        "emb": v,
        "cos": cos,
        "recompute_ok": rng.random(k) > 0.3,
        "label_consistency": rng.random(k),
        "donor": rng.choice([f"D{i}" for i in range(6)], size=k),
        "seasoned": rng.random(k) > 0.2,
    }


@pytest.mark.parametrize("policy", _policies(), ids=lambda p: p.name)
def test_defended_vote_matches_upstream(policy, rng):
    for _ in range(25):
        nb = _neighbourhood(rng)
        mine = defended_vote(nb["labels"], policy, recompute_ok=nb["recompute_ok"],
                             label_consistency=nb["label_consistency"], donor=nb["donor"],
                             seasoned=nb["seasoned"], cos=nb["cos"], emb=nb["emb"])
        theirs = up.defended_vote(nb["labels"], _as_upstream(policy),
                                  recompute_ok=nb["recompute_ok"],
                                  label_consistency=nb["label_consistency"],
                                  donor=nb["donor"], seasoned=nb["seasoned"],
                                  cos=nb["cos"], emb=nb["emb"])
        assert mine == theirs


@pytest.mark.parametrize("policy", _policies(), ids=lambda p: p.name)
def test_survivors_match_upstream(policy, rng):
    for _ in range(25):
        nb = _neighbourhood(rng)
        mine = survivors(nb["labels"], policy, recompute_ok=nb["recompute_ok"],
                         label_consistency=nb["label_consistency"], donor=nb["donor"],
                         seasoned=nb["seasoned"], emb=nb["emb"])
        theirs = up.survivors(nb["labels"], _as_upstream(policy),
                              recompute_ok=nb["recompute_ok"],
                              label_consistency=nb["label_consistency"],
                              donor=nb["donor"], seasoned=nb["seasoned"], emb=nb["emb"])
        np.testing.assert_array_equal(mine, theirs)


@pytest.mark.parametrize("policy", _policies(), ids=lambda p: p.name)
def test_policy_name_matches_upstream(policy):
    assert policy.name == _as_upstream(policy).name


@pytest.mark.parametrize("scheme", WEIGHT_SCHEMES)
def test_vote_weights_match_upstream(scheme, rng):
    for _ in range(50):
        k = int(rng.integers(1, 60))
        cos = np.sort(rng.uniform(-1.0, 1.0, size=k))[::-1]
        if rng.random() < 0.3:
            cos[0] = 1.0
        mine = vote_weights(cos, scheme)
        theirs = up.vote_weights(cos, up.DefenseConfig(weight=scheme))
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=0)


def test_vote_weights_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unknown weight scheme"):
        vote_weights(np.array([0.9]), "proximity")


def test_score_weight_is_clipped_not_infinite():
    """cos == 1.0 must give a large finite weight, not inf."""
    w = vote_weights(np.array([1.0]), "score")
    assert np.isfinite(w).all()
    assert w[0] == pytest.approx(1.0 / SCORE_EPS)


def test_dedup_keep_matches_upstream(rng):
    for tau in (0.5, 0.9, 0.99, 0.999, 1.0):
        for _ in range(30):
            k = int(rng.integers(1, 40))
            v = rng.normal(size=(k, 8)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            if k > 3:
                v[2] = v[1]
            np.testing.assert_array_equal(dedup_keep(v, tau), up.dedup_keep(v, tau))


def test_label_agreement_matches_upstream():
    cases = [
        ("A", ["A", "A", "B"]),
        ("A", []),
        ("A", ["B", "B"]),
        ("A", np.array(["A", "A", "A"])),
    ]
    for attached, nbrs in cases:
        assert label_agreement(attached, nbrs) == up.label_agreement(attached, nbrs)


def test_empty_neighbourhood_scores_one():
    """No evidence is not evidence against."""
    assert label_agreement("A", []) == 1.0


def test_admit_matches_upstream(rng):
    for policy in _policies():
        for _ in range(40):
            kw = {
                "recompute_ok": bool(rng.random() > 0.5),
                "label_consistency": float(rng.random()),
                "seasoned": bool(rng.random() > 0.5),
            }
            assert admit(policy, **kw) == up.admit(_as_upstream(policy), **kw)


@pytest.mark.parametrize(
    "policy,missing",
    [
        (VotePolicy(recompute=True), "recompute_ok"),
        (VotePolicy(label_consistency=True), "label_consistency"),
        (VotePolicy(donor_cap=1), "donor"),
        (VotePolicy(quarantine=True), "seasoned"),
        (VotePolicy(dedup_tau=0.99), "emb"),
        (VotePolicy(weight="score"), "cos"),
    ],
)
def test_missing_evidence_raises_rather_than_passing(policy, missing, rng):
    """A layer enabled without its evidence must raise, never silently pass."""
    nb = _neighbourhood(rng, k=8)
    kwargs = {k: nb[k] for k in
              ("recompute_ok", "label_consistency", "donor", "seasoned", "cos", "emb")}
    kwargs[missing] = None
    with pytest.raises(MissingEvidence):
        defended_vote(nb["labels"], policy, **kwargs)
    # and upstream raises too, for the same input
    with pytest.raises(ValueError):
        up.defended_vote(nb["labels"], _as_upstream(policy), **kwargs)


def test_everything_filtered_returns_none(rng):
    """An empty survivor set is an abstention, not a crash and not a label."""
    nb = _neighbourhood(rng, k=10)
    policy = VotePolicy(recompute=True)
    out = defended_vote(nb["labels"], policy,
                        recompute_ok=np.zeros(10, dtype=bool), cos=nb["cos"])
    assert out is None


def test_uniform_weighting_reproduces_majority_vote(rng):
    labels = np.array(["A", "A", "B"])
    cos = np.array([0.9, 0.8, 0.7])
    assert defended_vote(labels, VotePolicy(), cos=cos) == "A"


def test_dedup_tie_break_is_order_dependent():
    """The documented reason dedup is sound at ingest only.

    An exact copy sits at the same cosine as its original, so "keep the
    highest-ranked member" is a tie decided by array order. Listing the
    poison first keeps the poison.
    """
    v = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    genuine_first = np.array(["genuine", "poison"])
    poison_first = np.array(["poison", "genuine"])
    keep = dedup_keep(v, 0.999)
    assert keep.tolist() == [0]
    assert genuine_first[keep][0] == "genuine"
    assert poison_first[keep][0] == "poison"
