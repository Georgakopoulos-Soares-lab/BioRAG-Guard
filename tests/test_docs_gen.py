"""The layer reference is generated, so it cannot drift from the registry."""

from __future__ import annotations

from pathlib import Path

from biorag_guard.core.registry import all_layers
from biorag_guard.docs_gen import render_layer_reference

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "layers.md"


def test_every_layer_appears_in_the_reference():
    text = render_layer_reference()
    for name, cls in all_layers().items():
        assert f"## {name}" in text
        assert cls.spec.catches in text
        assert cls.spec.misses in text


def test_ingest_only_layers_are_marked():
    text = render_layer_reference()
    assert "**sound at ingest only**" in text


def test_corpus_wide_layers_say_a_sweep_is_required():
    text = render_layer_reference()
    assert "**a sweep only**" in text


def test_the_committed_reference_is_current():
    """Regenerate with `python -m biorag_guard.docs_gen` if this fails."""
    assert REFERENCE.exists(), "docs/layers.md has not been generated"
    assert REFERENCE.read_text() == render_layer_reference()
