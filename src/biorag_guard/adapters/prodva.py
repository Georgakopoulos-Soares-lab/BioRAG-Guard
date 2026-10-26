"""ProDVa: a text-keyed protein retrieval corpus.

The corpus is three files, and all three are attacker-writable:

1. the **data file** -- ``{"instruction", "sequence"}`` records whose
   ``instruction`` is embedded to build the retrieval index;
2. the **sequence map** -- the same pairs, read as a dict keyed by the
   description string, so two records sharing a description silently
   collapse to one;
3. the **fragment map** -- ``{"sequence", "phrases": [...]}``, keyed by
   sequence. A payload absent from this file contributes no fragments and
   so delivers nothing, which makes its presence a useful signal.

Upstream applies no normalisation, no dedup and no near-duplicate
filtering when records are added, so the set algebra across these three
files is both cheap and the first thing worth checking.

Embedding needs PubMedBERT. The adapter reads a precomputed
``vectors.npy`` when one is present and otherwise runs the text layers,
which need no model at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..core.arrays import Array
from ..core.finding import Finding
from ..core.record import Record, Severity, Verdict
from .base import CorpusAdapter, normalise

DATA_FILE = "data.jsonl"
SEQUENCE_MAP = "sequence_map.jsonl"
FRAGMENT_MAP = "fragment_map.jsonl"
VECTORS = "vectors.npy"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class ProDVaAdapter(CorpusAdapter):
    """Read a description-keyed protein corpus as three files."""

    name = "prodva"
    modality = "text-protein"

    def __init__(self) -> None:
        super().__init__()
        self._data: list[dict[str, Any]] = []
        self._seq_map: list[dict[str, Any]] = []
        self._frag_map: list[dict[str, Any]] = []
        self._vectors: Array | None = None

    @classmethod
    def sniff(cls, path: Path) -> float:
        path = Path(path)
        have = sum((path / f).exists() for f in (DATA_FILE, SEQUENCE_MAP, FRAGMENT_MAP))
        if have == 3:
            return 0.95
        if (path / DATA_FILE).exists() and have >= 2:
            return 0.7
        return 0.0

    def open(self, path: str | Path, **kwargs: Any) -> ProDVaAdapter:
        path = Path(path)
        self.path = path
        self._data = _read_jsonl(path / kwargs.get("data_file", DATA_FILE))
        self._seq_map = _read_jsonl(path / kwargs.get("sequence_map", SEQUENCE_MAP))
        self._frag_map = _read_jsonl(path / kwargs.get("fragment_map", FRAGMENT_MAP))
        if not self._data:
            raise FileNotFoundError(f"no records in {path / DATA_FILE}")
        vec = path / kwargs.get("vectors_file", VECTORS)
        if vec.exists():
            self._vectors = normalise(np.load(vec))
            if len(self._vectors) != len(self._data):
                raise ValueError(
                    f"{vec.name} has {len(self._vectors)} rows but the data file has "
                    f"{len(self._data)}"
                )
        self._opened = True
        return self

    # -- columns -----------------------------------------------------------

    @property
    def ids(self) -> Array:
        return np.array(
            [str(r.get("id", i)) for i, r in enumerate(self._data)], dtype=object
        )

    @property
    def texts(self) -> Array:
        """The descriptions. These are what the index is built from."""
        return np.array([r.get("instruction") for r in self._data], dtype=object)

    @property
    def payloads(self) -> Array:
        return np.array([r.get("sequence") for r in self._data], dtype=object)

    @property
    def vectors(self) -> Array | None:
        return self._vectors

    @property
    def labels(self) -> Array | None:
        return None

    def _fragments_for(self, sequence: str | None) -> list[str]:
        if sequence is None:
            return []
        return self._fragment_index().get(sequence, [])

    def _fragment_index(self) -> dict[str, list[str]]:
        if not hasattr(self, "_frag_idx"):
            idx: dict[str, list[str]] = {}
            for row in self._frag_map:
                seq = row.get("sequence")
                phrases = row.get("phrases", [])
                idx[str(seq)] = [
                    p.get("phrase", "") if isinstance(p, dict) else str(p) for p in phrases
                ]
            self._frag_idx = idx
        return self._frag_idx

    def _annotation_terms(self, sequence: str | None) -> list[str]:
        """Annotation names the corpus itself asserts for this sequence.

        These come from the fragment map's own ``name``/``description``
        fields. Nothing here derives biology from the sequence; the
        cross-modal layer checks two corpus assertions against each other.
        """
        if sequence is None:
            return []
        terms: list[str] = []
        for row in self._frag_map:
            if str(row.get("sequence")) != str(sequence):
                continue
            for p in row.get("phrases", []):
                if isinstance(p, dict):
                    for key in ("name", "description", "type"):
                        if p.get(key):
                            terms.append(str(p[key]).lower())
        return terms

    def iter_records(self, limit: int | None = None, since: Any = None) -> Iterator[Record]:
        n = len(self._data) if limit is None else min(limit, len(self._data))
        for i in range(n):
            row = self._data[i]
            seq = row.get("sequence")
            yield Record(
                id=str(row.get("id", i)),
                vector=None if self._vectors is None else self._vectors[i],
                text=row.get("instruction"),
                payload=seq,
                features={
                    "fragments": self._fragments_for(seq),
                    "annotation_terms": self._annotation_terms(seq),
                },
                source=row.get("source"),
            )

    def capabilities(self) -> set[str]:
        caps: set[str] = {"text", "payload"}
        if self._vectors is not None:
            caps.add("vector")
        return caps

    # -- integrity ------------------------------------------------------------

    def integrity(self) -> list[Finding]:
        return self.map_integrity()

    def map_integrity(self) -> list[Finding]:
        """Set algebra across the three attacker-writable files."""
        out: list[Finding] = []
        data_desc = [r.get("instruction") for r in self._data]
        seq_desc = {r.get("instruction") for r in self._seq_map}
        seq_seqs = {r.get("sequence") for r in self._seq_map}
        frag_seqs = set(self._fragment_index())

        # a description indexed twice silently collapses in the dict-keyed map
        seen: dict[str, int] = {}
        for d in data_desc:
            seen[str(d)] = seen.get(str(d), 0) + 1
        collided = {d: n for d, n in seen.items() if n > 1}
        if collided:
            total = sum(collided.values()) - len(collided)
            out.append(Finding(
                layer="crossmodal_maps", record_id="<corpus>", score=float(total),
                threshold=0.0, severity=Severity.HIGH, verdict=Verdict.REJECT,
                reason=(f"{len(collided):,} descriptions appear more than once across "
                        f"{total + len(collided):,} records. The sequence map is keyed by "
                        f"the description, so each collision silently drops a real record "
                        f"in favour of whichever was written last."),
                remedy="Make descriptions unique before indexing, or key the map on a "
                       "record id instead.",
                evidence={"n_collisions": len(collided),
                          "examples": list(collided)[:5]},
            ))

        missing_map = [d for d in data_desc if d not in seq_desc] if self._seq_map else []
        if missing_map:
            out.append(Finding(
                layer="crossmodal_maps", record_id="<corpus>", score=float(len(missing_map)),
                threshold=0.0, severity=Severity.HIGH, verdict=Verdict.REJECT,
                reason=(f"{len(missing_map):,} descriptions are in the index but absent "
                        f"from the description-to-sequence map, so retrieving them yields "
                        f"no protein."),
                remedy="Rebuild both files from one export.",
                evidence={"examples": [str(d)[:80] for d in missing_map[:5]]},
            ))

        if self._frag_map:
            no_frags = sorted(s for s in seq_seqs if s is not None and s not in frag_seqs)
            if no_frags:
                out.append(Finding(
                    layer="crossmodal_maps", record_id="<corpus>", score=float(len(no_frags)),
                    threshold=0.0, severity=Severity.MEDIUM, verdict=Verdict.FLAG,
                    reason=(f"{len(no_frags):,} sequences have no entry in the fragment "
                            f"map, so they contribute nothing to generation even when they "
                            f"are retrieved."),
                    remedy="Annotate them, or remove them from the corpus.",
                    evidence={"n": len(no_frags)},
                ))
            orphans = sorted(s for s in frag_seqs if s not in seq_seqs)
            if orphans:
                out.append(Finding(
                    layer="crossmodal_maps", record_id="<corpus>", score=float(len(orphans)),
                    threshold=0.0, severity=Severity.MEDIUM, verdict=Verdict.FLAG,
                    reason=(f"{len(orphans):,} entries in the fragment map refer to "
                            f"sequences that are in no record. Fragments can enter the "
                            f"generator's vocabulary through a sequence nothing points to."),
                    remedy="Drop the orphaned fragment entries.",
                    evidence={"n": len(orphans)},
                ))
        return out

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({
            "n_sequence_map": len(self._seq_map),
            "n_fragment_map": len(self._frag_map),
            "embeddings": self._vectors is not None,
        })
        return d
