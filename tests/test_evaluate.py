"""The evaluation harness, including its duty to report what it cannot do."""

from __future__ import annotations

import pytest

from biorag_guard import Guard, demo
from biorag_guard.evaluate import NOT_WIRED, evaluate_insert_arms, render_table


@pytest.fixture
def setup(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=50)
    guard = Guard.auto(corpus.path)
    guard.calibrate(budget=0.05)
    return guard, corpus


def test_every_insert_arm_is_evaluated(setup):
    guard, corpus = setup
    results = evaluate_insert_arms(guard, corpus, n=12)
    assert {r.arm for r in results} == {
        "relabel", "duplicate", "interpolate", "flood", "hub"
    }
    assert all(0.0 <= r.detection_rate <= 1.0 for r in results)
    assert all(r.n_poison == 12 for r in results)


def test_detection_is_reported_with_its_budget(setup):
    guard, corpus = setup
    results = evaluate_insert_arms(guard, corpus, n=10, budget=0.05)
    for r in results:
        d = r.to_dict()
        assert d["budget"] == 0.05
        assert "genuine_records_flagged" in d


def test_the_table_names_the_arms_it_cannot_run(setup):
    guard, corpus = setup
    table = render_table(evaluate_insert_arms(guard, corpus, n=5))
    assert "Arms not covered" in table
    for arm in NOT_WIRED:
        assert arm in table
    # the overwrite null must be stated, not omitted
    assert "manifest verify" in table


def test_per_layer_contribution_is_recorded(setup):
    guard, corpus = setup
    by_arm = {r.arm: r.layers for r in evaluate_insert_arms(guard, corpus, n=12)}
    # the measured asymmetry: recompute is silent on genuine-vector attacks
    assert "recompute" not in by_arm["relabel"]
    assert "recompute" not in by_arm["duplicate"]
    assert by_arm["interpolate"].get("recompute", 0) > 0
    assert by_arm["flood"].get("recompute", 0) > 0
