"""The integration contract: the two modes, the vote, and the API snippet."""

from __future__ import annotations

import numpy as np
import pytest

from biorag_guard import (
    Guard,
    Quarantined,
    Record,
    Verdict,
    demo,
)
from biorag_guard.core.baseline import Baseline
from biorag_guard.core.errors import BaselineMismatch, CorpusAlreadyContainsBatch
from fixtures import records_from


@pytest.fixture
def guard(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=60)
    g = Guard.auto(corpus.path)
    g.calibrate(budget=0.05)
    g.corpus = corpus  # type: ignore[attr-defined]
    return g


# -- the documented snippet ------------------------------------------------

def test_the_five_line_snippet_works_verbatim(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=40)
    guard = Guard.auto(corpus.path)
    guard.calibrate(budget=0.05)

    record = records_from(*demo.poison(corpus, "relabel", n=1))[0]
    result = guard.check(record)
    assert result.verdict is not Verdict.ADMIT
    assert result.reasons

    clean = next(guard.adapter.iter_records())
    clean = Record(id="brand_new", vector=clean.vector, label=clean.label,
                   donor=clean.donor)
    assert guard.check(clean).verdict in (Verdict.ADMIT, Verdict.FLAG,
                                          Verdict.QUARANTINE)


def test_check_one_record_returns_one_result(guard):
    rec = records_from(*demo.poison(guard.corpus, "flood", n=1))[0]
    out = guard.check(rec)
    assert out.record_id == rec.id


def test_gate_decorator_raises_on_a_failing_record(guard):
    added = []

    @guard.gate
    def add(record):
        added.append(record.id)
        return record.id

    bad = records_from(*demo.poison(guard.corpus, "duplicate", n=1))[0]
    with pytest.raises(Quarantined) as exc:
        add(bad)
    assert exc.value.record_id == bad.id
    assert exc.value.reasons
    assert added == []


def test_gating_context_manager_buffers_rejects(guard):
    class FakeIndex:
        def __init__(self):
            self.items = []

        def add(self, record):
            self.items.append(record.id)

    index = FakeIndex()
    batch = records_from(*demo.poison(guard.corpus, "relabel", n=5))
    with guard.gating(index) as safe:
        for rec in batch:
            safe.add(rec)
    assert index.items == []
    assert len(safe.rejected) == 5


# -- mode separation ------------------------------------------------------------

def test_check_refuses_a_batch_already_in_the_corpus(guard):
    """Scoring against a reference that holds the batch weakens the layer."""
    existing = list(guard.adapter.iter_records(limit=3))
    with pytest.raises(CorpusAlreadyContainsBatch) as exc:
        guard.check_many(existing)
    assert "BEFORE the batch" in str(exc.value)
    assert "allow_already_ingested" in str(exc.value)


def test_the_override_is_available_and_documented(guard):
    existing = list(guard.adapter.iter_records(limit=3))
    report = guard.check_many(existing, allow_already_ingested=True)
    assert len(report.results) == 3


def test_check_tells_the_user_what_it_cannot_see(guard):
    batch = records_from(*demo.poison(guard.corpus, "relabel", n=2))
    report = guard.check_many(batch)
    assert any("bioguard scan" in n for n in report.notes)
    assert any("bioguard scan" in s.reason for s in report.skipped)


def test_scan_states_its_sampling(guard):
    report = guard.scan(sample=50)
    assert any("Sampled 50" in n and "estimates" in n for n in report.notes)
    assert report.provenance["sample"] == 50


def test_scan_without_sampling_says_nothing_about_sampling(guard):
    report = guard.scan(sample=None)
    assert not any("Sampled" in n for n in report.notes)


def test_scan_since_filters_and_reports(guard):
    report = guard.scan(sample=None, since="2026-09-01")
    assert any("Restricted to" in n for n in report.notes)


# -- the defended vote ------------------------------------------------------------

def test_vote_returns_both_answers(guard):
    q = np.asarray(guard.adapter.vectors)[0]
    out = guard.vote(q)
    assert out.label is not None
    assert out.undefended_label is not None
    assert out.n_neighbours > 0


def test_donor_cap_reduces_survivors(guard):
    q = np.asarray(guard.adapter.vectors)[0]
    plain = guard.vote(q)
    guard.config.vote.donor_cap = 1
    capped = guard.vote(q)
    assert capped.n_survivors < plain.n_survivors
    assert "donorcap1" in capped.policy


def test_abstain_can_refuse(guard):
    q = np.asarray(guard.adapter.vectors)[0]
    guard.config.vote.abstain = True
    guard.config.vote.abstain_min_margin = 1.01  # impossible margin
    out = guard.vote(q)
    assert out.abstained is True and out.label is None


def test_vote_recompute_raises_rather_than_passing(guard):
    """The corpus has no counts for its own records; silence would be a lie."""
    guard.config.vote.recompute = True
    from biorag_guard.core.errors import MissingEvidence

    with pytest.raises(MissingEvidence, match="re-derived"):
        guard.vote(np.asarray(guard.adapter.vectors)[0])


# -- baseline handling ------------------------------------------------------------

def test_baseline_refuses_a_different_corpus(tmp_path):
    a = demo.build(tmp_path / "a", n_per_type=30)
    b = demo.build(tmp_path / "b", n_per_type=40, seed=99)
    ga = Guard.auto(a.path)
    baseline, _ = ga.calibrate(budget=0.05)
    gb = Guard.auto(b.path)
    with pytest.raises(BaselineMismatch, match="calibrate"):
        Guard(gb.adapter, gb.config, baseline)


def test_budget_retunes_without_refitting(guard):
    tight = guard.baseline.retune(0.01)
    loose = guard.baseline.retune(0.2)
    lc_t = tight.layers["label_consistency"].threshold
    lc_l = loose.layers["label_consistency"].threshold
    # lower score is suspicious, so a smaller budget means a lower threshold
    assert lc_t <= lc_l
    assert tight.layers["label_consistency"].budget == 0.01


def test_baseline_roundtrips(guard, tmp_path):
    path = tmp_path / "b.json"
    guard.baseline.save(path)
    again = Baseline.load(path)
    assert again.layers.keys() == guard.baseline.layers.keys()
    assert again.cost == guard.baseline.cost


def test_calibration_warns_about_an_already_poisoned_corpus(guard):
    assert any("absorbs the poison" in n for n in guard.baseline.notes)


def test_calibration_measures_the_cost_on_clean_data(guard):
    cost = guard.baseline.cost
    assert cost["available"] is True
    assert 0.0 <= cost["clean_accuracy"] <= 1.0
    assert 0.0 <= cost["abstention_rate"] <= 1.0
    assert "accuracy_delta" in cost


# -- quarantine ----------------------------------------------------------------------

def test_quarantine_writes_a_reversible_sidelist(guard, tmp_path):
    batch = records_from(*demo.poison(guard.corpus, "duplicate", n=4))
    report = guard.check_many(batch)
    path = guard.quarantine(report, batch)
    assert path is not None and path.exists()
    lines = [line for line in path.read_text().splitlines() if line]
    assert len(lines) == 4
    assert "reasons" in lines[0]


# -- explain -----------------------------------------------------------------------

def test_explain_reports_the_calibrated_threshold(guard):
    info = guard.explain("label_consistency")
    assert info["ported"] is True
    assert info["calibrated"]["threshold"] == guard.baseline.layers[
        "label_consistency"].threshold
    assert info["misses"]


def test_explain_marks_dedup_as_ingest_only(guard):
    assert guard.explain("dedup")["ingest_only"] is True


# -- policies ------------------------------------------------------------------------

def test_strict_policy_escalates_flags(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=40)
    guard = Guard.auto(corpus.path)
    guard.config.policy = "strict"
    guard.calibrate(budget=0.05)
    assert guard.config.verdict_for("qc_counts", Verdict.FLAG) is Verdict.QUARANTINE


def test_permissive_policy_still_stops_on_structural_findings(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=20)
    guard = Guard.auto(corpus.path)
    guard.config.policy = "permissive"
    assert guard.config.verdict_for("idx_meta", Verdict.REJECT) is Verdict.REJECT
    assert guard.config.verdict_for("label_consistency",
                                    Verdict.QUARANTINE) is Verdict.FLAG
