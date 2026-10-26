"""Report shape and the three rules the renderers must not break."""

from __future__ import annotations

import json

import pytest

from biorag_guard import Guard, demo
from biorag_guard.core.finding import Finding, Report
from biorag_guard.core.record import Severity, Verdict
from biorag_guard.report import render_html, render_terminal, write_report
from fixtures import records_from


@pytest.fixture
def report(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=40)
    guard = Guard.auto(corpus.path)
    guard.calibrate(budget=0.05)
    return guard.check_many(records_from(*demo.poison(corpus, "relabel", n=5)))


def test_json_is_versioned_and_round_trips(report):
    payload = json.loads(report.to_json())
    assert payload["schema_version"] == "1.0"
    for key in ("command", "verdict", "exit_code", "counts", "findings", "skipped",
                "retrieval", "outcome", "calibration", "provenance", "notes"):
        assert key in payload


def test_retrieval_and_outcome_are_separate_fields(report):
    payload = json.loads(report.to_json())
    assert "retrieval" in payload and "outcome" in payload
    assert payload["retrieval"] is not payload["outcome"]


def test_skipped_layers_are_always_reported(report):
    payload = json.loads(report.to_json())
    assert payload["skipped"], "a run with no skipped layers should still carry the key"
    assert all({"layer", "reason"} == set(s) for s in payload["skipped"])


def test_html_is_self_contained(report, tmp_path):
    html = render_html(report)
    assert html.startswith("<!doctype html>")
    for forbidden in ("http://", "https://", "<script"):
        assert forbidden not in html
    assert "Layers that did not run" in html


def test_terminal_output_leads_with_the_notes(report):
    text = render_terminal(report)
    # the note is wrapped to the terminal width, so compare on words
    flat = " ".join(text.split())
    assert "bioguard scan" in flat
    assert "verdict:" in text and "exit code" in text


def test_write_report_produces_both_formats(report, tmp_path):
    written = write_report(report, tmp_path / "out", ["html", "json"])
    assert {p.name for p in written} == {"report.html", "report.json"}
    assert all(p.stat().st_size > 0 for p in written)


def test_verdict_is_the_worst_seen():
    r = Report(command="check", corpus="x", adapter="generic")
    assert r.verdict is Verdict.ADMIT
    r.findings.append(Finding(
        layer="l", record_id="a", score=1.0, threshold=0.0, severity=Severity.LOW,
        verdict=Verdict.FLAG, reason="r", remedy="m",
    ))
    assert r.verdict is Verdict.FLAG and r.exit_code == 0
    r.findings.append(Finding(
        layer="l", record_id="b", score=1.0, threshold=0.0, severity=Severity.HIGH,
        verdict=Verdict.REJECT, reason="r", remedy="m",
    ))
    assert r.verdict is Verdict.REJECT and r.exit_code == 2


def test_numpy_values_survive_serialisation():
    import numpy as np

    r = Report(command="scan", corpus="x", adapter="generic")
    r.findings.append(Finding(
        layer="l", record_id="a", score=1.0, threshold=0.0, severity=Severity.LOW,
        verdict=Verdict.FLAG, reason="r", remedy="m",
        evidence={"ids": np.array([1, 2]), "n": np.int64(3), "frac": np.float32(0.5)},
    ))
    payload = json.loads(r.to_json())
    assert payload["findings"][0]["evidence"] == {"ids": [1, 2], "n": 3, "frac": 0.5}
