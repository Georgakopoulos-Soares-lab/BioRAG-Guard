"""The generic adapter: vectors in a ``.npy``, records in a ``.jsonl``.

This is the worked example for ``docs/extending.md`` and the backing store
for ``bioguard demo``. It is also the fastest way to put an unsupported
corpus behind BioRAG-Guard: export two files and every shared layer works.

Layout::

    mycorpus/
      vectors.npy     float32 (n, d); normalised on load
      records.jsonl   one JSON object per line, in the same order
                      {"id": ..., "label": ..., "donor": ..., "text": ...,
                       "payload": ..., "added_at": "2026-01-01T00:00:00",
                       "counts": [...], "features": {...}, "metadata": {...}}
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..core.finding import Finding
from ..core.record import Record, Severity, Verdict
from .base import CorpusAdapter, normalise

VECTORS = "vectors.npy"
RECORDS = "records.jsonl"


class GenericAdapter(CorpusAdapter):
    """A corpus held as a vector matrix plus a JSON-lines record file."""

    name = "generic"
    modality = "generic"

    def __init__(self) -> None:
        super().__init__()
        self._vectors: np.ndarray | None = None
        self._rows: list[dict[str, Any]] = []

    @classmethod
    def sniff(cls, path: Path) -> float:
        path = Path(path)
        if (path / VECTORS).exists() and (path / RECORDS).exists():
            return 0.9
        if (path / RECORDS).exists():
            return 0.4
        return 0.0

    def open(self, path: str | Path, **kwargs: Any) -> GenericAdapter:
        path = Path(path)
        self.path = path
        vec_file = path / kwargs.get("vectors_file", VECTORS)
        rec_file = path / kwargs.get("records_file", RECORDS)
        if not rec_file.exists():
            raise FileNotFoundError(f"{rec_file} not found")
        self._rows = [json.loads(line) for line in rec_file.read_text().splitlines() if line]
        if vec_file.exists():
            self._vectors = normalise(np.load(vec_file))
            if len(self._vectors) != len(self._rows):
                raise ValueError(
                    f"{vec_file.name} has {len(self._vectors)} rows but "
                    f"{rec_file.name} has {len(self._rows)}"
                )
        self._opened = True
        return self

    # -- columns --------------------------------------------------------

    @property
    def ids(self) -> np.ndarray:
        return np.array([str(r.get("id", i)) for i, r in enumerate(self._rows)], dtype=object)

    @property
    def vectors(self) -> np.ndarray | None:
        return self._vectors

    @property
    def labels(self) -> np.ndarray | None:
        if not self._rows or not any("label" in r for r in self._rows):
            return None
        return np.array([r.get("label") for r in self._rows], dtype=object)

    @property
    def donors(self) -> np.ndarray | None:
        if not self._rows or not any("donor" in r for r in self._rows):
            return None
        return np.array([r.get("donor") for r in self._rows], dtype=object)

    # -- records ---------------------------------------------------------

    def iter_records(
        self, limit: int | None = None, since: datetime | None = None
    ) -> Iterator[Record]:
        n = len(self._rows) if limit is None else min(limit, len(self._rows))
        for i in range(n):
            rec = self._to_record(i)
            if since is not None and (rec.added_at is None or rec.added_at < since):
                continue
            yield rec

    def _to_record(self, i: int) -> Record:
        row = self._rows[i]
        added = row.get("added_at")
        counts = row.get("counts")
        return Record(
            id=str(row.get("id", i)),
            vector=None if self._vectors is None else self._vectors[i],
            label=row.get("label"),
            text=row.get("text"),
            payload=row.get("payload"),
            counts=None if counts is None else np.asarray(counts),
            donor=row.get("donor"),
            added_at=None if added is None else datetime.fromisoformat(added),
            features=row.get("features", {}),
            metadata=row.get("metadata", {}),
            source=row.get("source"),
        )

    def recompute(self, records: Sequence[Record]) -> np.ndarray | None:
        """Re-derive vectors from counts, if the corpus carries them.

        The generic corpus has no encoder, so the derivation is the
        documented identity: normalise the counts. It exists so the layer
        can be exercised end to end; a real adapter re-runs its model.
        """
        if not records or any(r.counts is None for r in records):
            return None
        return normalise(np.vstack([np.asarray(r.counts, dtype=np.float32) for r in records]))

    def capabilities(self) -> set[str]:
        caps = super().capabilities()
        if any(r.get("counts") is not None for r in self._rows):
            caps.add("counts")
        if any(r.get("text") is not None for r in self._rows):
            caps.add("text")
        if any(r.get("payload") is not None for r in self._rows):
            caps.add("payload")
        if any(r.get("added_at") is not None for r in self._rows):
            caps.add("added_at")
        return caps

    def integrity(self) -> list[Finding]:
        """Duplicate ids, and a vector/record count mismatch."""
        out: list[Finding] = []
        ids = [str(r.get("id", i)) for i, r in enumerate(self._rows)]
        seen: dict[str, int] = {}
        for rid in ids:
            seen[rid] = seen.get(rid, 0) + 1
        for rid, n in seen.items():
            if n > 1:
                out.append(
                    Finding(
                        layer="idx_meta", record_id=rid, score=float(n), threshold=1.0,
                        severity=Severity.HIGH, verdict=Verdict.REJECT,
                        reason=(f"The id {rid!r} appears {n} times. A repeated key means one "
                                "record silently overwrites another."),
                        remedy="Make ids unique before indexing; check the export that built "
                               "this corpus.",
                        evidence={"occurrences": n},
                    )
                )
        return out
