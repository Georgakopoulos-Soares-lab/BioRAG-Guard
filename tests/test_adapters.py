"""Adapters, against directories laid out the way the real corpora are.

The SCimilarity and ProDVa fixtures are small, but their structure is the
released structure: the same file names, the same column conventions, the
same three-file split. What is not covered is the model-backed path --
re-encoding counts needs the weights -- and the tests say so by name
rather than by omission.
"""

from __future__ import annotations

import numpy as np
import pytest

from biorag_guard import Guard, demo
from biorag_guard.adapters import available, open_corpus, sniff_all
from biorag_guard.adapters.prodva import ProDVaAdapter
from biorag_guard.adapters.scimilarity import SCimilarityAdapter
from biorag_guard.core.errors import AdapterNotFound
from fixtures import write_prodva_corpus, write_scimilarity_corpus

# -- discovery ---------------------------------------------------------

def test_every_builtin_adapter_imports():
    assert set(available()) >= {"generic", "scimilarity", "prodva"}


def test_sniffing_picks_the_right_adapter(tmp_path):
    sci = write_scimilarity_corpus(tmp_path / "sci")
    pro = write_prodva_corpus(tmp_path / "pro")
    gen = demo.build(tmp_path / "gen", n_per_type=5).path
    assert sniff_all(sci)[0][0] == "scimilarity"
    assert sniff_all(pro)[0][0] == "prodva"
    assert sniff_all(gen)[0][0] == "generic"


def test_unrecognised_path_names_what_was_tried(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(AdapterNotFound) as exc:
        open_corpus(empty)
    message = str(exc.value)
    for adapter in ("generic", "scimilarity", "prodva"):
        assert adapter in message
    assert "vectors.npy" in message


# -- SCimilarity ---------------------------------------------------------

def test_scimilarity_reads_the_released_layout(tmp_path):
    root = write_scimilarity_corpus(tmp_path / "sci", n=120)
    adapter = SCimilarityAdapter().open(root)
    d = adapter.describe()
    assert d["n_records"] == 120
    assert d["label_column"] == "celltype_name"
    assert d["donor_column"] == "donor_id"
    assert adapter.labels is not None and adapter.donors is not None
    assert adapter.ids[0] == "cell00000"


def test_scimilarity_works_without_the_hnsw_index(tmp_path):
    """Everything but `recompute` needs only the embeddings and metadata."""
    root = write_scimilarity_corpus(tmp_path / "sci", n=80)
    adapter = SCimilarityAdapter().open(root)
    assert adapter.describe()["search"] == "exact"
    idx, cos = adapter.neighbors(np.asarray(adapter.vectors)[:3], k=5)
    assert idx.shape == (3, 5)
    assert np.all(cos[:, 0] >= cos[:, -1])


def test_scimilarity_detects_an_index_metadata_mismatch(tmp_path):
    root = write_scimilarity_corpus(tmp_path / "sci", n=60, truncate_meta=4)
    adapter = SCimilarityAdapter().open(root)
    findings = adapter.integrity()
    assert any("metadata holds" in f.reason for f in findings)


def test_scimilarity_detects_non_unique_ids(tmp_path):
    """Upstream does not guarantee the index column is unique."""
    root = write_scimilarity_corpus(tmp_path / "sci", n=60, duplicate_ids=3)
    adapter = SCimilarityAdapter().open(root)
    findings = adapter.integrity()
    assert any("appear more than once" in f.reason for f in findings)


def test_scimilarity_recompute_is_unavailable_without_the_model(tmp_path):
    """Not applicable must be reported, never silently passed."""
    root = write_scimilarity_corpus(tmp_path / "sci", n=20)
    adapter = SCimilarityAdapter().open(root)
    assert adapter.recompute(list(adapter.iter_records(limit=2))) is None


def test_scimilarity_runs_end_to_end(tmp_path):
    root = write_scimilarity_corpus(tmp_path / "sci", n=150)
    guard = Guard.auto(root)
    baseline, _ = guard.calibrate(budget=0.05)
    assert "label_consistency" in baseline.layers
    report = guard.scan(sample=None)
    assert report.adapter == "scimilarity"
    # recompute has no counts here and must say so rather than pass
    assert any(s.layer == "recompute" for s in report.skipped)


# -- ProDVa -----------------------------------------------------------------

def test_prodva_reads_three_files(tmp_path):
    root = write_prodva_corpus(tmp_path / "pro", n=20)
    adapter = ProDVaAdapter().open(root)
    d = adapter.describe()
    assert d["n_records"] == 20
    assert d["n_sequence_map"] == 20
    assert d["n_fragment_map"] == 20
    assert adapter.capabilities() == {"text", "payload"}


def test_prodva_carries_annotation_terms_from_the_fragment_map(tmp_path):
    root = write_prodva_corpus(tmp_path / "pro", n=8)
    adapter = ProDVaAdapter().open(root)
    rec = next(adapter.iter_records())
    assert rec.features["annotation_terms"]
    assert rec.features["fragments"]


def test_prodva_flags_colliding_description_keys(tmp_path):
    """The map is keyed by description, so a collision drops a real record."""
    root = write_prodva_corpus(tmp_path / "pro", n=20, collide=3)
    findings = ProDVaAdapter().open(root).map_integrity()
    assert any("silently drops a real record" in f.reason for f in findings)


def test_prodva_flags_descriptions_missing_from_the_sequence_map(tmp_path):
    root = write_prodva_corpus(tmp_path / "pro", n=20, missing_from_map=2)
    findings = ProDVaAdapter().open(root).map_integrity()
    assert any("absent" in f.reason and "sequence map" in f.reason for f in findings)


def test_prodva_flags_orphaned_fragment_entries(tmp_path):
    root = write_prodva_corpus(tmp_path / "pro", n=10, orphan_fragments=2)
    findings = ProDVaAdapter().open(root).map_integrity()
    assert any("refer to" in f.reason for f in findings)


def test_prodva_clean_corpus_has_no_map_findings(tmp_path):
    root = write_prodva_corpus(tmp_path / "pro", n=15)
    assert ProDVaAdapter().open(root).map_integrity() == []


def test_prodva_text_layers_run_without_any_model(tmp_path):
    root = write_prodva_corpus(tmp_path / "pro", n=24)
    guard = Guard.auto(root)
    baseline, _ = guard.calibrate(budget=0.1)
    assert "text_nearduplicate" in baseline.layers
    report = guard.scan(sample=None)
    assert report.adapter == "prodva"
    # no embeddings in this fixture, so vector layers must be skipped aloud
    assert any(s.layer == "hub" for s in report.skipped)


def test_prodva_near_duplicate_catches_a_copied_description(tmp_path):
    root = write_prodva_corpus(tmp_path / "pro", n=24)
    guard = Guard.auto(root)
    guard.calibrate(budget=0.1)
    from biorag_guard import Record

    original = next(guard.adapter.iter_records())
    copy = Record(id="poison", text=original.text, payload="MKV" * 20,
                  features={"annotation_terms": ["unrelated transporter"],
                            "fragments": ["MKV" * 5]})
    report = guard.check_many([copy])
    fired = report.by_layer()
    assert fired.get("text_nearduplicate", 0) == 1


# -- the generic adapter -------------------------------------------------------

def test_generic_detects_duplicate_ids(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=10)
    rows = list(corpus.rows)
    rows[3]["id"] = rows[0]["id"]
    demo.write_batch(corpus.path, corpus.vectors, rows)
    adapter = open_corpus(corpus.path)
    assert any("appears 2 times" in f.reason for f in adapter.integrity())


def test_generic_rejects_a_length_mismatch(tmp_path):
    corpus = demo.build(tmp_path / "c", n_per_type=10)
    np.save(corpus.path / "vectors.npy", np.asarray(corpus.vectors)[:-2])
    with pytest.raises(ValueError, match="rows"):
        open_corpus(corpus.path)
