"""The manifest, and the asymmetry that justifies it.

The research measured that over a 7,913,892-record reference an index
hash does not detect an in-place relabel while a label hash does. These
tests assert that property of the implementation directly, because it is
the single claim the tool's second job rests on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biorag_guard import Guard, demo
from biorag_guard.core.manifest import Manifest, compare


@pytest.fixture
def corpus(tmp_path: Path):
    return demo.build(tmp_path / "c", n_per_type=30)


def test_index_hash_is_blind_to_a_relabel(corpus, tmp_path):
    """The headline asymmetry, as a test."""
    guard = Guard.auto(corpus.path)
    before = guard.adapter.manifest()
    demo.relabel_in_place(corpus, n=10)
    after = Guard.auto(corpus.path).adapter.manifest()
    diff = compare(before, after)

    assert diff.index_changed is False, "the vectors did not move"
    assert diff.labels_changed is True
    assert diff.relabel_in_place is True
    assert diff.n_labels_changed == 10


def test_changed_records_are_named_not_merely_counted(corpus, tmp_path):
    guard = Guard.auto(corpus.path)
    before = guard.adapter.manifest()
    changed = set(demo.relabel_in_place(corpus, n=7))
    after = Guard.auto(corpus.path).adapter.manifest()
    diff = compare(before, after)
    assert set(diff.changed_label_ids) == changed
    assert all(t == "fibroblast" for t in diff.changed_label_to)
    assert diff.destination_tally()["fibroblast"] == 7


def test_roundtrip_through_disk_keeps_the_label_column(corpus, tmp_path):
    path = tmp_path / "m.json"
    corpus_guard = Guard.auto(corpus.path)
    corpus_guard.adapter.manifest().save(path)
    assert (tmp_path / "m.labels.npz").exists()
    loaded = Manifest.load(path)
    assert loaded.labels is not None
    np.testing.assert_array_equal(
        np.asarray(loaded.labels, dtype=str),
        np.asarray(corpus_guard.adapter.labels, dtype=str),
    )


def test_diff_joins_on_id_so_an_insertion_is_not_a_mass_relabel(corpus, tmp_path):
    """A record added in the middle must not report everything after it as changed."""
    guard = Guard.auto(corpus.path)
    before = guard.adapter.manifest()
    rows = list(corpus.rows)
    vecs = np.asarray(corpus.vectors)
    rows.insert(5, {**rows[5], "id": "inserted_0001"})
    vecs = np.insert(vecs, 5, vecs[5], axis=0)
    demo.write_batch(corpus.path, vecs, rows)
    after = Guard.auto(corpus.path).adapter.manifest()
    diff = compare(before, after)
    assert diff.n_labels_changed == 0
    assert diff.index_changed is True


def test_manifest_without_labels_says_so_rather_than_guessing(corpus, tmp_path):
    guard = Guard.auto(corpus.path)
    before = guard.adapter.manifest(keep_labels=False)
    path = tmp_path / "m.json"
    before.save(path)
    demo.relabel_in_place(corpus, n=4)
    after = Guard.auto(corpus.path).adapter.manifest()
    diff = compare(Manifest.load(path), after)
    assert diff.labels_changed is True
    assert diff.n_labels_changed == 0
    assert any("kept no label column" in n for n in diff.notes)


def test_clean_corpus_compares_clean(corpus):
    guard = Guard.auto(corpus.path)
    diff = compare(guard.adapter.manifest(), guard.adapter.manifest())
    assert diff.clean is True
    assert diff.relabel_in_place is False
