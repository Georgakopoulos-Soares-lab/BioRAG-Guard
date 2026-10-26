"""A generated config must be loadable. It is the file users inherit."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from biorag_guard import Guard, demo
from biorag_guard.cli.app import app
from biorag_guard.core.config import POLICIES, GuardConfig

runner = CliRunner()


def test_init_output_loads_back_and_opens_the_corpus(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=20)
    assert runner.invoke(app, ["init", str(corpus.path)]).exit_code == 0

    guard = Guard.load(corpus.path / "bioguard.yaml")
    assert len(guard.adapter) == 80
    assert guard.config.layers
    assert guard.config.budget == 0.05


def test_config_survives_a_save_load_cycle(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=10)
    cfg = GuardConfig.for_corpus(corpus.path, "generic")
    cfg.budget = 0.01
    cfg.policy = "strict"
    cfg.vote.donor_cap = 2
    cfg.vote.abstain = True
    path = tmp_path / "bioguard.yaml"
    cfg.save(path)

    again = GuardConfig.load(path)
    assert again.budget == 0.01
    assert again.policy == "strict"
    assert again.vote.donor_cap == 2
    assert again.vote.abstain is True
    assert again.path.resolve() == corpus.path.resolve()


def test_an_unknown_policy_is_rejected_by_name(tmp_path):
    path = tmp_path / "bioguard.yaml"
    path.write_text("policy: paranoid\n")
    with pytest.raises(ValueError, match="unknown policy"):
        GuardConfig.load(path)


@pytest.mark.parametrize("policy", POLICIES)
def test_every_named_policy_loads(tmp_path, policy):
    corpus = demo.build(tmp_path / "c", n_per_type=10)
    cfg = GuardConfig.for_corpus(corpus.path, "generic")
    cfg.policy = policy
    path = tmp_path / "bioguard.yaml"
    cfg.save(path)
    assert Guard.load(path).config.policy == policy


def test_a_layer_can_be_switched_off_in_the_config(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=20)
    runner.invoke(app, ["init", str(corpus.path)])
    path = corpus.path / "bioguard.yaml"
    text = path.read_text().replace(
        "  dedup:\n    enabled: true", "  dedup:\n    enabled: false"
    )
    path.write_text(text)
    guard = Guard.load(path)
    assert "dedup" not in guard._enabled_names()
