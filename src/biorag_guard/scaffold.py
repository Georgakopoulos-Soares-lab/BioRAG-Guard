"""Scaffold a new adapter, so the second one someone writes takes an afternoon."""

from __future__ import annotations

from pathlib import Path

_ADAPTER = '''"""Adapter for the {title} corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from biorag_guard.adapters.base import CorpusAdapter, normalise
from biorag_guard.core.arrays import Array
from biorag_guard.core.finding import Finding


class {cls}(CorpusAdapter):
    """Read a {title} corpus.

    Similarity is cosine, descending. Vectors are unit-norm float32.
    Memory-map anything large rather than loading it into RAM.
    """

    name = "{name}"
    modality = "cell"          # or "text-protein", or your own

    def __init__(self) -> None:
        super().__init__()
        self._vectors: Array | None = None

    @classmethod
    def sniff(cls, path: Path) -> float:
        """Confidence in [0, 1] that this path holds a {title} corpus."""
        return 0.9 if (Path(path) / "REPLACE_ME").exists() else 0.0

    def open(self, path: str | Path, **kwargs: Any) -> "{cls}":
        self.path = Path(path)
        # self._vectors = normalise(np.load(self.path / "...", mmap_mode="r"))
        raise NotImplementedError("implement open()")

    @property
    def ids(self) -> Array:
        raise NotImplementedError("implement ids")

    @property
    def vectors(self) -> Array | None:
        return self._vectors

    @property
    def labels(self) -> Array | None:
        """The label each record CLAIMS, or None if the corpus has none."""
        return None

    def recompute(self, records: Any) -> Array | None:
        """Re-derive embeddings from the raw inputs.

        Return None when the corpus cannot. The layer then reports itself as
        not applicable instead of passing every record, which is the
        behaviour the rest of the package depends on.
        """
        return None

    def integrity(self) -> list[Finding]:
        """Structural checks only this corpus can make: row counts, key uniqueness."""
        return []
'''

_TEST = '''"""Tests for the {name} adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from {module} import {cls}


def build_fixture(root: Path) -> Path:
    """Write a small corpus in the REAL layout, not a mock.

    Testing against the genuine structure is what makes the integrity checks
    mean anything.
    """
    root.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("write a minimal real-layout corpus here")


@pytest.mark.skip(reason="implement build_fixture first")
def test_sniff_recognises_the_layout(tmp_path):
    root = build_fixture(tmp_path / "corpus")
    assert {cls}.sniff(root) > 0.5
    assert {cls}.sniff(tmp_path) == 0.0


@pytest.mark.skip(reason="implement build_fixture first")
def test_a_planted_record_is_caught_and_a_clean_one_is_not(tmp_path):
    from biorag_guard import Guard

    root = build_fixture(tmp_path / "corpus")
    guard = Guard.auto(root)
    guard.calibrate(budget=0.05)
    raise NotImplementedError("assert on a planted record and a clean one")


@pytest.mark.skip(reason="implement build_fixture first")
def test_what_this_corpus_cannot_do_is_reported(tmp_path):
    """Anything unavailable must appear in report.skipped, never be silent."""
    raise NotImplementedError
'''


def scaffold_adapter(name: str, out: Path | str = ".") -> list[Path]:
    """Write an adapter stub and its test. Returns the paths written."""
    out = Path(out)
    slug = name.strip().lower().replace("-", "_")
    cls = "".join(part.capitalize() for part in slug.split("_")) + "Adapter"
    title = slug.replace("_", " ")
    module = f"{slug}_adapter"

    adapter_path = out / f"{module}.py"
    test_path = out / f"test_{module}.py"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(_ADAPTER.format(cls=cls, name=slug, title=title))
    test_path.write_text(_TEST.format(cls=cls, name=slug, module=module))
    return [adapter_path, test_path]
