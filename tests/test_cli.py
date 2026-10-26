"""The command line, including the exit codes CI depends on."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from biorag_guard import demo
from biorag_guard.cli.app import app

runner = CliRunner()


@pytest.fixture
def corpus(tmp_path):
    return demo.build(tmp_path / "c", n_per_type=40)


def run(*args: str):
    return runner.invoke(app, list(args))


def test_demo_runs_end_to_end_with_no_data(tmp_path):
    """The adoption lever: no downloads, no weights, under a minute."""
    result = run("demo", "--workdir", str(tmp_path / "d"))
    assert result.exit_code == 0, result.output
    for marker in ("building a synthetic corpus", "taking a manifest", "calibrating",
                   "guarding at ingest", "guarding on demand"):
        assert marker in result.output
    assert "index hash   unchanged" in result.output
    assert "label hash   CHANGED" in result.output


def test_init_writes_a_working_config(corpus):
    result = run("init", str(corpus.path))
    assert result.exit_code == 0, result.output
    cfg = corpus.path / "bioguard.yaml"
    assert cfg.exists()
    text = cfg.read_text()
    assert text.startswith("# BioRAG-Guard configuration")
    assert "budget" in text and "access" in text
    assert "bioguard manifest create" in result.output


def test_init_refuses_an_unrecognised_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run("init", str(empty))
    assert result.exit_code == 3


def test_calibrate_prints_thresholds_and_cost(corpus, tmp_path):
    result = run("calibrate", str(corpus.path), "--budget", "0.05",
                 "--out", str(tmp_path / "b.json"))
    assert result.exit_code == 0, result.output
    assert "sacrifice at most 5.00%" in result.output
    assert "What the vote policy costs on clean data" in result.output
    assert (tmp_path / "b.json").exists()


def test_scan_exit_code_is_zero_on_a_clean_corpus(corpus, tmp_path):
    run("calibrate", str(corpus.path), "--out", str(corpus.path / "baseline.json"))
    result = run("scan", str(corpus.path), "--sample", "40")
    assert result.exit_code in (0, 1)
    assert "Sampled 40" in result.output


def test_check_exit_code_signals_a_quarantine(corpus, tmp_path):
    run("calibrate", str(corpus.path), "--out", str(corpus.path / "baseline.json"))
    vecs, rows = demo.poison(corpus, "duplicate", n=6)
    batch = demo.write_batch(tmp_path / "batch", vecs, rows)
    result = run("check", str(batch), "--corpus", str(corpus.path))
    assert result.exit_code == 1, result.output
    assert "quarantine" in result.output


def test_check_json_output_is_valid_and_versioned(corpus, tmp_path):
    run("calibrate", str(corpus.path), "--out", str(corpus.path / "baseline.json"))
    vecs, rows = demo.poison(corpus, "relabel", n=3)
    batch = demo.write_batch(tmp_path / "batch", vecs, rows)
    result = run("check", str(batch), "--corpus", str(corpus.path), "--json")
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["command"] == "check"
    assert len(payload["results"]) == 3


def test_manifest_create_then_verify_detects_a_relabel(corpus, tmp_path):
    m = tmp_path / "m.json"
    assert run("manifest", "create", str(corpus.path), "--out", str(m)).exit_code == 0
    clean = run("manifest", "verify", str(corpus.path), "--manifest", str(m))
    assert clean.exit_code == 0
    assert "label hash      unchanged" in clean.output

    demo.relabel_in_place(corpus, n=5)
    dirty = run("manifest", "verify", str(corpus.path), "--manifest", str(m))
    assert dirty.exit_code == 2
    assert "index hash      unchanged" in dirty.output
    assert "label hash      CHANGED" in dirty.output
    assert "5 records changed label" in dirty.output
    assert "would have reported this corpus as clean" in dirty.output


def test_explain_lists_every_layer():
    result = run("explain")
    assert result.exit_code == 0
    for name in ("recompute", "label_consistency", "dedup", "manifest_label", "hub"):
        assert name in result.output


def test_explain_a_layer_states_what_it_misses():
    result = run("explain", "recompute")
    assert result.exit_code == 0
    assert "misses" in result.output
    assert "relabel" in result.output.lower()


def test_explain_shows_the_calibrated_threshold(corpus):
    run("calibrate", str(corpus.path), "--out", str(corpus.path / "baseline.json"))
    result = run("explain", "label_consistency", "--corpus", str(corpus.path))
    assert "on this corpus" in result.output
    assert "budget" in result.output


def test_version():
    result = run("version")
    assert result.exit_code == 0 and "biorag-guard" in result.output
