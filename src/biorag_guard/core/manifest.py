"""The manifest: a record of corpus state, so an edit becomes visible.

This is the cheapest high-value thing in the package, and the measurements
are the reason it exists. Against an attacker who overwrites records in
place, every vote-time layer fails -- in the research grid, relabelling
existing records left attack success at 0.149-0.178 whether the stack was
on or off. What catches it is arithmetic over the label column:

======================  ==========================  ==========
check                   catches a relabel in place  wall time
======================  ==========================  ==========
hash of the index       **no**                      1.37 s
hash of the labels      yes                         0.10 s
element-wise label diff yes                         0.31 s
======================  ==========================  ==========

(over a 7,913,892-record reference; ``03b_04_both_integrity``.)

A checksum over the index cannot see a relabel, because the vectors do not
move. So the manifest hashes the label column separately from the vectors,
and -- because knowing *that* something changed is much less useful than
knowing *which* records -- it keeps the label column itself, stored as
categorical codes beside the JSON.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_VERSION = "1.0"
#: Records per block hash. Localises a change without storing every digest.
BLOCK = 100_000


def _digest(chunks: Iterator[bytes]) -> str:
    h = hashlib.blake2b(digest_size=32)
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def hash_array(arr: np.ndarray, block: int = BLOCK) -> tuple[str, list[str]]:
    """Whole-array digest plus one digest per block of rows."""
    arr = np.ascontiguousarray(arr)
    blocks: list[str] = []
    whole = hashlib.blake2b(digest_size=32)
    for lo in range(0, len(arr), block):
        piece = np.ascontiguousarray(arr[lo:lo + block])
        raw = piece.tobytes()
        whole.update(raw)
        blocks.append(hashlib.blake2b(raw, digest_size=16).hexdigest())
    return whole.hexdigest(), blocks


def hash_strings(values: np.ndarray, block: int = BLOCK) -> tuple[str, list[str]]:
    """Digest of a string column, block-wise. Encoding is pinned to UTF-8."""
    vals = np.asarray(values, dtype=object)
    blocks: list[str] = []
    whole = hashlib.blake2b(digest_size=32)
    for lo in range(0, len(vals), block):
        raw = "\n".join("" if v is None else str(v) for v in vals[lo:lo + block]).encode()
        whole.update(raw)
        blocks.append(hashlib.blake2b(raw, digest_size=16).hexdigest())
    return whole.hexdigest(), blocks


@dataclass
class Manifest:
    """What the corpus looked like when the manifest was taken."""

    adapter: str
    corpus: str
    n_records: int
    created_at: str
    index_hash: str | None = None
    index_blocks: list[str] = field(default_factory=list)
    label_hash: str | None = None
    label_blocks: list[str] = field(default_factory=list)
    id_hash: str | None = None
    dims: int | None = None
    notes: list[str] = field(default_factory=list)
    version: str = MANIFEST_VERSION
    #: path of the sidecar holding the label column, relative to the json
    labels_file: str | None = None
    _labels: np.ndarray | None = field(default=None, repr=False, compare=False)
    _ids: np.ndarray | None = field(default=None, repr=False, compare=False)

    # -- construction ---------------------------------------------------

    @classmethod
    def build(
        cls,
        adapter_name: str,
        corpus: str,
        ids: np.ndarray,
        vectors: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        keep_labels: bool = True,
    ) -> Manifest:
        """Hash a corpus. ``keep_labels`` stores the column for element-wise diff."""
        ids = np.asarray(ids, dtype=object)
        m = cls(
            adapter=adapter_name,
            corpus=str(corpus),
            n_records=int(len(ids)),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        m.id_hash, _ = hash_strings(ids)
        m._ids = ids
        if vectors is not None:
            m.dims = int(np.asarray(vectors).shape[1])
            m.index_hash, m.index_blocks = hash_array(np.asarray(vectors))
        if labels is not None:
            labels = np.asarray(labels, dtype=object)
            m.label_hash, m.label_blocks = hash_strings(labels)
            if keep_labels:
                m._labels = labels
        return m

    # -- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._labels is not None:
            side = path.with_suffix(".labels.npz")
            codes, cats = _categorise(self._labels)
            ids = np.asarray(self._ids, dtype=str) if self._ids is not None else np.array([])
            np.savez_compressed(side, codes=codes, categories=cats, ids=ids)
            self.labels_file = side.name
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        path = Path(path)
        data = json.loads(path.read_text())
        data.pop("version", None)
        m = cls(version=MANIFEST_VERSION, **data)
        if m.labels_file:
            side = path.parent / m.labels_file
            if side.exists():
                z = np.load(side, allow_pickle=False)
                m._labels = np.asarray(z["categories"], dtype=object)[z["codes"]]
                m._ids = np.asarray(z["ids"], dtype=object)
            else:
                m.notes.append(f"label sidecar {m.labels_file} is missing")
        return m

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "adapter": self.adapter,
            "corpus": self.corpus,
            "n_records": self.n_records,
            "created_at": self.created_at,
            "dims": self.dims,
            "index_hash": self.index_hash,
            "index_blocks": self.index_blocks,
            "label_hash": self.label_hash,
            "label_blocks": self.label_blocks,
            "id_hash": self.id_hash,
            "labels_file": self.labels_file,
            "notes": self.notes,
        }

    # -- comparison -----------------------------------------------------

    @property
    def labels(self) -> np.ndarray | None:
        return self._labels

    @property
    def ids(self) -> np.ndarray | None:
        return self._ids


@dataclass
class ManifestDiff:
    """What changed between a stored manifest and the corpus as it is now."""

    n_before: int
    n_after: int
    index_changed: bool
    labels_changed: bool
    ids_changed: bool
    changed_label_ids: list[str] = field(default_factory=list)
    changed_label_from: list[str] = field(default_factory=list)
    changed_label_to: list[str] = field(default_factory=list)
    changed_index_blocks: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_labels_changed(self) -> int:
        return len(self.changed_label_ids)

    @property
    def clean(self) -> bool:
        return not (self.index_changed or self.labels_changed or self.ids_changed)

    @property
    def relabel_in_place(self) -> bool:
        """The case no vote-time layer catches: labels moved, vectors did not."""
        return self.labels_changed and not self.index_changed

    def destination_tally(self) -> dict[str, int]:
        """Which labels the changed records moved TO, commonest first."""
        tally: dict[str, int] = {}
        for lab in self.changed_label_to:
            tally[lab] = tally.get(lab, 0) + 1
        return dict(sorted(tally.items(), key=lambda kv: -kv[1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_before": self.n_before,
            "n_after": self.n_after,
            "index_changed": self.index_changed,
            "labels_changed": self.labels_changed,
            "ids_changed": self.ids_changed,
            "n_labels_changed": self.n_labels_changed,
            "relabel_in_place": self.relabel_in_place,
            "changed_index_blocks": self.changed_index_blocks,
            "destination_tally": self.destination_tally(),
            "examples": [
                {"id": i, "from": f, "to": t}
                for i, f, t in list(
                    zip(self.changed_label_ids, self.changed_label_from,
                        self.changed_label_to)
                )[:20]
            ],
            "notes": self.notes,
        }


def compare(before: Manifest, after: Manifest) -> ManifestDiff:
    """Compare two manifests, localising label changes to individual records."""
    diff = ManifestDiff(
        n_before=before.n_records,
        n_after=after.n_records,
        index_changed=bool(
            before.index_hash is not None
            and after.index_hash is not None
            and before.index_hash != after.index_hash
        ),
        labels_changed=bool(
            before.label_hash is not None
            and after.label_hash is not None
            and before.label_hash != after.label_hash
        ),
        ids_changed=bool(
            before.id_hash is not None
            and after.id_hash is not None
            and before.id_hash != after.id_hash
        ),
    )
    if before.index_hash is None or after.index_hash is None:
        diff.notes.append("no index hash on one side; vector changes were not checked")
    if before.label_hash is None or after.label_hash is None:
        diff.notes.append("no label hash on one side; relabels were not checked")

    for i, (b, a) in enumerate(zip(before.index_blocks, after.index_blocks)):
        if b != a:
            diff.changed_index_blocks.append(i)

    if diff.labels_changed:
        if before.labels is None:
            diff.notes.append(
                "the stored manifest kept no label column, so the changed records "
                "could not be named; rebuild the manifest with keep_labels=True"
            )
        elif after.labels is None:
            diff.notes.append("the corpus exposed no label column to compare against")
        else:
            diff_ids = _elementwise_label_diff(before, after)
            diff.changed_label_ids, diff.changed_label_from, diff.changed_label_to = diff_ids
    return diff


def _elementwise_label_diff(
    before: Manifest, after: Manifest
) -> tuple[list[str], list[str], list[str]]:
    """Which records kept their id and changed their label.

    Joins on id rather than on position, so an insertion in the middle of
    the corpus does not report every record after it as relabelled.
    """
    b_ids, b_lab = before.ids, before.labels
    a_ids, a_lab = after.ids, after.labels
    assert b_lab is not None and a_lab is not None
    if b_ids is None or a_ids is None:
        n = min(len(b_lab), len(a_lab))
        moved = np.flatnonzero(np.asarray(b_lab[:n]) != np.asarray(a_lab[:n]))
        return ([str(i) for i in moved], [str(b_lab[i]) for i in moved],
                [str(a_lab[i]) for i in moved])

    lookup = {str(k): i for i, k in enumerate(np.asarray(b_ids, dtype=object))}
    ids_out: list[str] = []
    from_out: list[str] = []
    to_out: list[str] = []
    for j, rid in enumerate(np.asarray(a_ids, dtype=object)):
        i = lookup.get(str(rid))
        if i is None:
            continue
        if str(b_lab[i]) != str(a_lab[j]):
            ids_out.append(str(rid))
            from_out.append(str(b_lab[i]))
            to_out.append(str(a_lab[j]))
    return ids_out, from_out, to_out


def _categorise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cats, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    return codes.astype(np.int32), cats
