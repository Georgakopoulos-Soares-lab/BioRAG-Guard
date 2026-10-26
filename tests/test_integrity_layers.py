"""The manifest layers, through the layer interface the scan actually uses.

These two carry the result the second mode exists for, so they are tested
directly rather than only through the CLI: `manifest_index` must stay
silent on a relabel, and `manifest_label` must name the records.
"""

from __future__ import annotations

import numpy as np
import pytest

from biorag_guard import Guard, Verdict, demo
from biorag_guard.core.registry import Context, get_layer


@pytest.fixture
def corpus(tmp_path):
    return demo.build(tmp_path / "c", n_per_type=30)


def _ctx(path):
    return Context(adapter=Guard.auto(path).adapter)


def test_index_layer_is_silent_on_a_relabel(corpus, tmp_path):
    """A checksum over the vectors cannot see a label change. By design."""
    m = tmp_path / "m.json"
    Guard.auto(corpus.path).manifest_create(m)
    demo.relabel_in_place(corpus, n=8)

    index_layer = get_layer("manifest_index")(path=str(m))
    label_layer = get_layer("manifest_label")(path=str(m))
    ctx = _ctx(corpus.path)

    assert index_layer.run([], ctx) == []
    found = label_layer.run([], ctx)
    assert len(found) == 1
    assert found[0].verdict is Verdict.REJECT
    assert found[0].score == 8.0


def test_label_layer_says_the_index_hash_would_have_passed(corpus, tmp_path):
    m = tmp_path / "m.json"
    Guard.auto(corpus.path).manifest_create(m)
    demo.relabel_in_place(corpus, n=5)
    found = get_layer("manifest_label")(path=str(m)).run([], _ctx(corpus.path))
    assert "reported this corpus as clean" in found[0].reason
    assert "fibroblast" in found[0].reason


def test_index_layer_fires_when_vectors_move(corpus, tmp_path):
    m = tmp_path / "m.json"
    Guard.auto(corpus.path).manifest_create(m)
    vecs = np.asarray(corpus.vectors).copy()
    vecs[3] = vecs[4]
    demo.write_batch(corpus.path, vecs, corpus.rows)

    found = get_layer("manifest_index")(path=str(m)).run([], _ctx(corpus.path))
    assert len(found) == 1
    assert found[0].verdict is Verdict.REJECT
    assert "stored vectors have changed" in found[0].reason


def test_both_layers_are_silent_on_an_untouched_corpus(corpus, tmp_path):
    m = tmp_path / "m.json"
    Guard.auto(corpus.path).manifest_create(m)
    ctx = _ctx(corpus.path)
    assert get_layer("manifest_index")(path=str(m)).run([], ctx) == []
    assert get_layer("manifest_label")(path=str(m)).run([], ctx) == []


def test_a_missing_manifest_is_reported_not_passed(corpus, tmp_path):
    """No manifest means not applicable, with the command that fixes it."""
    layer = get_layer("manifest_label")(path=str(tmp_path / "absent.json"))
    with pytest.raises(NotImplementedError, match="bioguard manifest create"):
        layer.run([], _ctx(corpus.path))


def test_scan_surfaces_the_relabel_end_to_end(corpus, tmp_path):
    guard = Guard.auto(corpus.path)
    guard.manifest_create()
    guard.calibrate(budget=0.05)
    demo.relabel_in_place(corpus, n=12)

    after = Guard.auto(corpus.path)
    after.calibrate(budget=0.05)
    report = after.scan(sample=None)
    layers = report.by_layer()
    assert layers.get("manifest_label") == 1
    assert "manifest_index" not in layers
    assert report.verdict is Verdict.REJECT
    assert report.exit_code == 2
