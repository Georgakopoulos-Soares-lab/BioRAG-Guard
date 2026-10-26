"""The scaffold must produce code that at least parses and imports cleanly."""

from __future__ import annotations

import ast
import subprocess
import sys

from biorag_guard.scaffold import scaffold_adapter


def test_scaffold_writes_valid_python(tmp_path):
    written = scaffold_adapter("my-atlas", tmp_path)
    assert [p.name for p in written] == ["my_atlas_adapter.py", "test_my_atlas_adapter.py"]
    for path in written:
        ast.parse(path.read_text())


def test_scaffolded_adapter_subclasses_the_protocol(tmp_path):
    adapter_path, _ = scaffold_adapter("demo_corpus", tmp_path)
    source = adapter_path.read_text()
    assert "class DemoCorpusAdapter(CorpusAdapter):" in source
    assert 'name = "demo_corpus"' in source
    # the three things a new adapter must not get wrong
    assert "cosine, descending" in source
    assert "Return None when the corpus cannot" in source


def test_scaffolded_tests_are_skipped_not_failing(tmp_path):
    _, test_path = scaffold_adapter("demo_corpus", tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert "3 skipped" in result.stdout, result.stdout
