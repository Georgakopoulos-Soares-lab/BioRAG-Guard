"""SCimilarity: a cell-embedding reference atlas.

The attack surface here is four things: the vectors in the kNN index, the
rows in the metadata, the alignment between the two, and the counts
upstream of the encoder. A poisoned cell wins by being *returned* -- there
is no generation step -- so hubness and label coherence carry most of the
weight, and the integrity of the index/metadata join carries the rest.

**The adapter works without torch.** Everything except ``recompute`` needs
only the precomputed embeddings and the metadata, both of which ship with
the model directory. Installing the model and its weights is required only
to re-derive embeddings from counts.

Expected layout (``model_v1.1`` as released)::

    model_dir/
      cellsearch/
        full_kNN.bin            hnswlib index
        full_kNN_meta.csv       metadata, one row per vector
        full_kNN_embedding.npy  the embeddings themselves
      cell_metadata/            TileDB array (optional, read if present)

Upstream's own note is that the metadata build is not hardened and the
index column is not guaranteed unique, so :meth:`integrity` checks exactly
that.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.finding import Finding
from ..core.record import Record, Severity, Verdict
from .base import CorpusAdapter, exact_neighbors, normalise

CELLSEARCH = "cellsearch"
EMBEDDING = "full_kNN_embedding.npy"
METADATA = "full_kNN_meta.csv"
KNN_INDEX = "full_kNN.bin"

#: Columns to try, in order, for the label a record claims.
LABEL_COLUMNS = ("celltype_name", "cell_type", "celltype", "prediction", "author_label")
#: Columns to try, in order, for the provenance group ``donor_cap`` uses.
DONOR_COLUMNS = ("donor_id", "donor", "sample", "sample_id", "study")
ID_COLUMNS = ("index", "cell_id", "soma_joinid", "barcode")


class SCimilarityAdapter(CorpusAdapter):
    """Read a SCimilarity model directory as a corpus."""

    name = "scimilarity"
    modality = "cell"

    def __init__(self) -> None:
        super().__init__()
        self._vectors: np.ndarray | None = None
        self._meta: Any = None
        self._ids: np.ndarray | None = None
        self._label_column: str | None = None
        self._donor_column: str | None = None
        self._knn: Any = None
        self._knn_size: int | None = None
        self._notes: list[str] = []

    # -- discovery -------------------------------------------------------

    @classmethod
    def sniff(cls, path: Path) -> float:
        path = Path(path)
        cs = path / CELLSEARCH
        if not cs.is_dir():
            return 0.0
        have = sum(
            (cs / f).exists() for f in (EMBEDDING, METADATA, KNN_INDEX)
        )
        if have == 0:
            return 0.0
        if (cs / EMBEDDING).exists() and (cs / METADATA).exists():
            return 0.95
        return 0.5 + 0.1 * have

    def open(self, path: str | Path, **kwargs: Any) -> SCimilarityAdapter:
        import pandas as pd

        path = Path(path)
        self.path = path
        cs = path / kwargs.get("cellsearch_dir", CELLSEARCH)
        emb_file = cs / kwargs.get("embedding_file", EMBEDDING)
        meta_file = cs / kwargs.get("metadata_file", METADATA)
        knn_file = cs / kwargs.get("knn_file", KNN_INDEX)

        if not emb_file.exists():
            raise FileNotFoundError(
                f"{emb_file} not found. BioRAG-Guard reads the precomputed embeddings "
                "rather than re-encoding the atlas; without them only the manifest "
                "layers can run."
            )
        self._vectors = normalise(np.load(emb_file, mmap_mode="r"))

        if meta_file.exists():
            self._meta = pd.read_csv(meta_file)
            self._label_column = kwargs.get("label_column") or _first_present(
                self._meta.columns, LABEL_COLUMNS
            )
            self._donor_column = kwargs.get("donor_column") or _first_present(
                self._meta.columns, DONOR_COLUMNS
            )
            id_col = kwargs.get("id_column") or _first_present(
                self._meta.columns, ID_COLUMNS
            )
            if id_col is not None:
                self._ids = np.asarray(self._meta[id_col], dtype=object)
            if self._label_column is None:
                self._notes.append(
                    f"no cell-type column found among {LABEL_COLUMNS}; label-based "
                    "layers will report themselves as not applicable"
                )
        else:
            self._notes.append(f"{meta_file.name} is missing; the corpus has no metadata")

        if knn_file.exists():
            self._knn_size = knn_file.stat().st_size
            self._load_knn(knn_file)
        self._opened = True
        return self

    def _load_knn(self, knn_file: Path) -> None:
        """Use the shipped hnswlib index when hnswlib is installed.

        Falling back to exact search is not a silent downgrade: it is
        slower but returns the same or better neighbours, and the report
        records which path ran.
        """
        try:
            import hnswlib
        except ImportError:
            self._notes.append(
                "hnswlib is not installed, so searches use exact cosine over the "
                "embeddings instead of the shipped graph "
                "(pip install 'biorag-guard[scimilarity]')"
            )
            return
        try:
            assert self._vectors is not None
            index = hnswlib.Index(space="cosine", dim=int(self._vectors.shape[1]))
            index.load_index(str(knn_file))
            self._knn = index
        except Exception as exc:  # pragma: no cover - depends on the shipped file
            self._notes.append(f"could not load {knn_file.name} ({exc}); using exact search")

    # -- columns -----------------------------------------------------------

    @property
    def ids(self) -> np.ndarray:
        if self._ids is not None:
            return self._ids
        n = 0 if self._vectors is None else len(self._vectors)
        return np.array([str(i) for i in range(n)], dtype=object)

    @property
    def vectors(self) -> np.ndarray | None:
        return self._vectors

    @property
    def labels(self) -> np.ndarray | None:
        if self._meta is None or self._label_column is None:
            return None
        return np.asarray(self._meta[self._label_column], dtype=object)

    @property
    def donors(self) -> np.ndarray | None:
        if self._meta is None or self._donor_column is None:
            return None
        return np.asarray(self._meta[self._donor_column], dtype=object)

    # -- search ---------------------------------------------------------------

    def neighbors(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._knn is None:
            return exact_neighbors(np.asarray(vectors), np.asarray(self._vectors), k)
        q = np.asarray(vectors, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        k = int(min(k, self._knn.get_current_count()))
        self._knn.set_ef(max(k, 50))
        idx, dist = self._knn.knn_query(q, k=k)
        # hnswlib 'cosine' space returns 1 - cos
        return np.asarray(idx, dtype=np.int64), 1.0 - np.asarray(dist, dtype=np.float64)

    def recompute(self, records: Sequence[Record]) -> np.ndarray | None:
        """Re-encode from counts with the SCimilarity model, if it is installed.

        Returns None -- which makes the layer report itself as not
        applicable -- when the model, its weights, or the counts are
        absent. The released reference does not ship the count matrices,
        which is why the research could not even time this step.
        """
        if not records or any(r.counts is None for r in records):
            return None
        try:
            from scimilarity import CellEmbedding
        except ImportError:
            return None
        if self._embedder is None:  # pragma: no cover - needs weights
            self._embedder = CellEmbedding(model_path=str(self.path))
        counts = np.vstack([np.asarray(r.counts, dtype=np.float32) for r in records])
        return normalise(self._embedder.get_embeddings(counts))  # pragma: no cover

    _embedder: Any = None

    # -- integrity --------------------------------------------------------------

    def integrity(self) -> list[Finding]:
        """Row counts across the files, and uniqueness of the index column."""
        out: list[Finding] = []
        n_emb = 0 if self._vectors is None else len(self._vectors)
        n_meta = 0 if self._meta is None else len(self._meta)
        if self._meta is not None and n_emb != n_meta:
            out.append(Finding(
                layer="idx_meta", record_id="<corpus>", score=abs(n_emb - n_meta),
                threshold=0.0, severity=Severity.HIGH, verdict=Verdict.REJECT,
                reason=(f"The embedding file holds {n_emb:,} vectors but the metadata holds "
                        f"{n_meta:,} rows. Every answer from this corpus joins the two by "
                        f"position, so the labels are attached to the wrong cells."),
                remedy="Rebuild the metadata from the same export that produced the "
                       "embeddings.",
                evidence={"n_embeddings": n_emb, "n_metadata": n_meta},
            ))
        if self._ids is not None:
            uniq, counts = np.unique(np.asarray(self._ids, dtype=str), return_counts=True)
            dup = uniq[counts > 1]
            if dup.size:
                out.append(Finding(
                    layer="idx_meta", record_id="<corpus>", score=float(dup.size),
                    threshold=0.0, severity=Severity.HIGH, verdict=Verdict.REJECT,
                    reason=(f"{dup.size:,} record ids appear more than once. Upstream does "
                            f"not guarantee this column is unique, and a repeated key means "
                            f"one cell silently stands in for another."),
                    remedy="Re-key the metadata on a column that is unique per cell.",
                    evidence={"n_duplicate_ids": int(dup.size),
                              "examples": [str(x) for x in dup[:10]]},
                ))
        return out

    def capabilities(self) -> set[str]:
        caps = super().capabilities()
        return caps

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({
            "label_column": self._label_column,
            "donor_column": self._donor_column,
            "search": "hnswlib" if self._knn is not None else "exact",
            "notes": list(self._notes),
        })
        return d


def _first_present(columns: Any, candidates: Sequence[str]) -> str | None:
    have = {str(c).lower(): str(c) for c in columns}
    for want in candidates:
        if want in have:
            return have[want]
    return None
