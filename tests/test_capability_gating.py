"""A layer the corpus cannot feed must be skipped with a plain reason.

This is the "say so in the report" rule, and it is easy to break in a way
nothing else catches: the layer runs, fails somewhere deep, and the user
sees an implementation error where they should have seen one sentence
about their corpus.
"""

from __future__ import annotations

from biorag_guard import Guard, demo
from fixtures import write_prodva_corpus, write_scimilarity_corpus


def _reason(report, layer: str) -> str:
    matches = [s.reason for s in report.skipped if s.layer == layer]
    assert matches, f"{layer} was neither run nor reported as skipped"
    return matches[0]


def test_counts_layers_are_skipped_on_a_corpus_without_counts(tmp_path):
    """A SCimilarity reference ships embeddings, not count matrices."""
    root = write_scimilarity_corpus(tmp_path / "sci", n=120)
    guard = Guard.auto(root)
    _, report = guard.calibrate(budget=0.05)

    for layer in ("qc_counts", "recompute"):
        reason = _reason(report, layer)
        assert "no finite scores" not in reason, (
            f"{layer} reported an implementation failure, not a corpus fact: {reason}"
        )
    assert "carries no counts" in _reason(report, "qc_counts")


def test_vector_layers_are_skipped_on_a_corpus_without_embeddings(tmp_path):
    """A ProDVa corpus with no precomputed vectors still runs its text layers."""
    root = write_prodva_corpus(tmp_path / "pro", n=20)
    guard = Guard.auto(root)
    _, report = guard.calibrate(budget=0.1)
    assert "carries no vector" in _reason(report, "hub")


def test_label_layers_are_skipped_on_an_unlabelled_corpus(tmp_path):
    root = write_prodva_corpus(tmp_path / "pro", n=20)
    guard = Guard.auto(root)
    report = guard.scan(sample=None)
    assert "manifest_label" not in [f.layer for f in report.findings]
    assert "carries no label" in _reason(report, "manifest_label")


def test_no_skip_reason_is_an_exception_string(tmp_path):
    """Skip reasons are sentences about the corpus, not tracebacks."""
    corpus = demo.build(tmp_path / "c", n_per_type=20)
    guard = Guard.auto(corpus.path)
    _, report = guard.calibrate(budget=0.05)
    for skipped in report.skipped:
        assert "Traceback" not in skipped.reason
        assert not skipped.reason.startswith(skipped.layer + ":"), (
            f"{skipped.layer} leaked an internal error message: {skipped.reason}"
        )
