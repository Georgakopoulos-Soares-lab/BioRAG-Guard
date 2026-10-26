"""The adapter boundary: how BioRAG-Guard reads one kind of corpus.

A new corpus should be about a hundred lines. Everything a layer needs goes
through this protocol, so a correct adapter makes every shared layer
available immediately and switches on the modality-specific ones through
:meth:`CorpusAdapter.capabilities`.

Similarity is **cosine, descending** throughout. An adapter whose index is
L2 converts; it does not leak the convention upward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..core.arrays import Array
from ..core.finding import Finding
from ..core.manifest import Manifest
from ..core.record import Record


class CorpusAdapter(ABC):
    """Read-only view of one corpus.

    Subclasses set :attr:`name` and :attr:`modality`, implement
    :meth:`sniff` and :meth:`open`, and expose the corpus through
    :attr:`ids` / :attr:`vectors` / :attr:`labels` plus whichever optional
    hooks apply.
    """

    name: str = "base"
    modality: str = "unknown"

    def __init__(self) -> None:
        self.path: Path | None = None
        self._opened = False

    # -- discovery ------------------------------------------------------

    @classmethod
    @abstractmethod
    def sniff(cls, path: Path) -> float:
        """Confidence in [0, 1] that ``path`` is this kind of corpus."""

    @abstractmethod
    def open(self, path: str | Path, **kwargs: Any) -> CorpusAdapter:
        """Open the corpus. Returns self so calls chain."""

    # -- the corpus itself ----------------------------------------------

    @property
    @abstractmethod
    def ids(self) -> Array:
        """(n,) record ids, in index order."""

    @property
    def vectors(self) -> Array | None:
        """(n, d) unit-norm embeddings in index order, or None."""
        return None

    @property
    def labels(self) -> Array | None:
        """(n,) the label each record CLAIMS, or None."""
        return None

    @property
    def donors(self) -> Array | None:
        """(n,) provenance group per record, or None."""
        return None

    def __len__(self) -> int:
        return int(len(self.ids))

    # -- records ---------------------------------------------------------

    def iter_records(
        self, limit: int | None = None, since: datetime | None = None
    ) -> Iterator[Record]:
        """Walk the corpus. The default builds records from the columns."""
        ids = self.ids
        vecs = self.vectors
        labs = self.labels
        dons = self.donors
        n = len(ids) if limit is None else min(limit, len(ids))
        for i in range(n):
            yield Record(
                id=str(ids[i]),
                vector=None if vecs is None else np.asarray(vecs[i]),
                label=None if labs is None else str(labs[i]),
                donor=None if dons is None else str(dons[i]),
            )

    def get(self, ids: Sequence[str]) -> list[Record]:
        want = {str(i) for i in ids}
        return [r for r in self.iter_records() if r.id in want]

    # -- model-backed operations ------------------------------------------

    def embed(self, records: Sequence[Record]) -> Array:
        """Embed records that arrive without a vector."""
        raise NotImplementedError(f"{self.name} cannot embed records")

    def recompute(self, records: Sequence[Record]) -> Array | None:
        """Re-derive embeddings from raw counts, for the ``recompute`` layer.

        Returns None when the corpus cannot do it -- no model, or no counts.
        A None here makes the layer report itself as not applicable rather
        than silently passing every record.
        """
        return None

    # -- search -----------------------------------------------------------

    def neighbors(self, vectors: Array, k: int) -> tuple[Array, Array]:
        """Top-k neighbours of each query.

        Returns:
            ``(idx, cos)``, each (n, k'), cosine descending. k' may be less
            than k on a small corpus.
        """
        ref = self.vectors
        if ref is None:
            raise NotImplementedError(f"{self.name} exposes no vectors to search")
        return exact_neighbors(np.asarray(vectors), np.asarray(ref), k)

    def probe_queries(self, n: int, seed: int = 0) -> Array:
        """Queries used to measure what the corpus RETURNS.

        The default samples existing records, which is the right probe for
        a reference atlas: it asks what a realistic query would retrieve.
        """
        ref = self.vectors
        if ref is None:
            raise NotImplementedError(f"{self.name} exposes no vectors to probe with")
        rng = np.random.default_rng(seed)
        take = min(n, len(ref))
        pick = rng.choice(len(ref), size=take, replace=False)
        return np.asarray(ref)[np.sort(pick)]

    # -- integrity ---------------------------------------------------------

    def manifest(self, keep_labels: bool = True) -> Manifest:
        """Hash the corpus as it stands now."""
        return Manifest.build(
            adapter_name=self.name,
            corpus=str(self.path),
            ids=self.ids,
            vectors=self.vectors,
            labels=self.labels,
            keep_labels=keep_labels,
        )

    def integrity(self) -> list[Finding]:
        """Structural checks that need no baseline and no manifest."""
        return []

    # -- capability declaration ---------------------------------------------

    def capabilities(self) -> set[str]:
        """What this corpus can supply. Layers needing more are skipped, loudly."""
        caps = {"vector"} if self.vectors is not None else set()
        if self.labels is not None:
            caps.add("label")
        if self.donors is not None:
            caps.add("donor")
        return caps

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "modality": self.modality,
            "path": str(self.path),
            "n_records": len(self),
            "dims": None if self.vectors is None else int(self.vectors.shape[1]),
            "capabilities": sorted(self.capabilities()),
        }


def exact_neighbors(
    query: Array, ref: Array, k: int, block_size: int = 1024
) -> tuple[Array, Array]:
    """Blocked exact cosine search. Deterministic, and the fallback everywhere."""
    query = np.asarray(query, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if query.ndim == 1:
        query = query[None, :]
    k = int(min(k, len(ref)))
    if k == 0:
        return np.zeros((len(query), 0), dtype=np.int64), np.zeros((len(query), 0))
    idx_out = np.empty((len(query), k), dtype=np.int64)
    cos_out = np.empty((len(query), k), dtype=np.float64)
    for lo in range(0, len(query), block_size):
        hi = min(lo + block_size, len(query))
        cos = query[lo:hi] @ ref.T
        part = np.argpartition(-cos, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(cos, part, axis=1)
        order = np.argsort(-vals, axis=1, kind="stable")
        idx_out[lo:hi] = np.take_along_axis(part, order, axis=1)
        cos_out[lo:hi] = np.take_along_axis(vals, order, axis=1)
    return idx_out, cos_out


def normalise(vectors: Array) -> Array:
    """Unit-norm rows, float32. A zero row stays zero rather than becoming NaN."""
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return np.asarray(v / norm, dtype=np.float32)
