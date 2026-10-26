"""A tiny synthetic corpus, so the whole flow runs in seconds with no data.

No downloads, no model weights. The corpus is four well-separated cell
types in a low-dimensional space with a donor structure, which is enough
for every shared layer to do something real, and the poison builder
reproduces the shapes of the measured attack taxonomy:

``relabel``      a genuine record resubmitted under another type's label
``duplicate``    verbatim copies of genuine records, relabelled
``interpolate``  vectors synthesised between source and target
``flood``        a tight cluster of synthesised records near one anchor
``hub``          records positioned to be returned for everything

These are *defence fixtures*. They exist so the layers can be tested and
demonstrated against the attack shapes the research measured; the
construction is deliberately naive and is not an attack implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

CELL_TYPES = ("T cell", "B cell", "monocyte", "fibroblast")
DIM = 16


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return np.asarray(v / n, dtype=np.float32)


@dataclass
class DemoCorpus:
    """A generated corpus, in memory and on disk."""

    path: Path
    vectors: np.ndarray
    rows: list[dict[str, Any]]

    @property
    def labels(self) -> np.ndarray:
        return np.array([r["label"] for r in self.rows], dtype=object)


def build(
    path: str | Path,
    n_per_type: int = 150,
    n_donors: int = 6,
    seed: int = 7,
    spread: float = 0.22,
) -> DemoCorpus:
    """Write a clean generic-layout corpus to ``path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Non-negative, like expression: a count vector then reproduces its own
    # embedding exactly under the corpus's (identity) encoder, so `recompute`
    # passes every genuine record and has something real to catch on
    # synthesised geometry.
    centres = _unit(np.abs(rng.normal(size=(len(CELL_TYPES), DIM))))
    start = datetime(2026, 1, 1)

    vectors, rows = [], []
    for t, name in enumerate(CELL_TYPES):
        for i in range(n_per_type):
            v = _unit(np.abs(centres[t] + spread * rng.normal(size=DIM)))[0]
            counts = v * rng.integers(800, 4000)
            rows.append({
                "id": f"cell_{t:02d}_{i:04d}",
                "label": name,
                "donor": f"donor_{(i % n_donors):02d}",
                "added_at": (start + timedelta(days=int(i % 90))).isoformat(),
                "counts": np.round(counts, 3).tolist(),
                "source": f"study_{t:02d}",
            })
            vectors.append(v)
    vecs = np.vstack(vectors).astype(np.float32)
    _write(path, vecs, rows)
    return DemoCorpus(path=path, vectors=vecs, rows=rows)


def _write(path: Path, vectors: np.ndarray, rows: list[dict[str, Any]]) -> None:
    np.save(path / "vectors.npy", vectors)
    with (path / "records.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def poison(
    corpus: DemoCorpus,
    kind: str,
    n: int = 20,
    source: str = "T cell",
    target: str = "fibroblast",
    seed: int = 11,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build a batch in the shape of one measured attack. Returns (vectors, rows).

    The batch is returned, never written into the corpus: these fixtures
    are for testing defences, and a poisoned corpus is not something this
    package produces on disk.
    """
    rng = np.random.default_rng(seed)
    vecs = np.asarray(corpus.vectors)
    labels = corpus.labels
    src_idx = np.flatnonzero(labels == source)
    tgt_idx = np.flatnonzero(labels == target)
    if src_idx.size == 0 or tgt_idx.size == 0:
        raise ValueError(f"corpus has no {source!r} or no {target!r}")
    pick = rng.choice(src_idx, size=min(n, src_idx.size), replace=False)
    now = datetime(2026, 10, 1)
    out_v, out_r = [], []

    for j, i in enumerate(pick):
        # The counts always come from a real record, because an attacker
        # submitting a cell submits measurements too. Whether those counts
        # still reproduce the vector is exactly what separates the two
        # defence families: a genuine vector re-derives and `recompute` sees
        # nothing, a synthesised one does not.
        source_row = int(i)
        if kind == "relabel":
            v = vecs[i].copy()
        elif kind == "duplicate":
            source_row = int(pick[0])
            v = vecs[source_row].copy()
        elif kind == "interpolate":
            alpha = rng.uniform(0.3, 0.9)
            other = vecs[rng.choice(tgt_idx)]
            v = _unit((1 - alpha) * vecs[i] + alpha * other)[0]
        elif kind == "flood":
            anchor = vecs[src_idx[0]]
            v = _unit(np.abs(anchor + 0.08 * rng.normal(size=vecs.shape[1])))[0]
        elif kind == "hub":
            v = _unit(np.abs(vecs.mean(axis=0) + 0.01 * rng.normal(size=vecs.shape[1])))[0]
        else:
            raise ValueError(f"unknown poison kind {kind!r}")
        out_v.append(v)
        out_r.append({
            "id": f"poison_{kind}_{j:03d}",
            "label": target,
            "donor": "donor_attacker",
            "added_at": now.isoformat(),
            "source": "submission_2026-10-01",
            "counts": list(corpus.rows[source_row]["counts"]),
        })
    return np.vstack(out_v).astype(np.float32), out_r


def write_batch(path: str | Path, vectors: np.ndarray,
                rows: list[dict[str, Any]]) -> Path:
    """Write a batch in the generic layout, for `bioguard check <dir>`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    _write(path, np.asarray(vectors, dtype=np.float32), rows)
    return path


def relabel_in_place(
    corpus: DemoCorpus, n: int = 30, source: str = "T cell",
    target: str = "fibroblast", seed: int = 13,
) -> list[str]:
    """Rewrite labels on records already in the corpus, leaving vectors alone.

    This is the overwrite arm, and the reason the manifest exists: the
    index is byte-identical afterwards, so a checksum over the vectors
    reports the corpus as clean.
    """
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(corpus.labels == source)
    pick = rng.choice(idx, size=min(n, idx.size), replace=False)
    for i in pick:
        corpus.rows[int(i)]["label"] = target
    _write(corpus.path, corpus.vectors, corpus.rows)
    return [corpus.rows[int(i)]["id"] for i in pick]
