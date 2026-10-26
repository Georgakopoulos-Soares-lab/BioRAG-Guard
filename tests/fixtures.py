"""Corpus fixtures that every integration test shares."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from biorag_guard import Record


def records_from(vectors: np.ndarray, rows: list[dict]) -> list[Record]:
    out = []
    for i, r in enumerate(rows):
        counts = r.get("counts")
        added = r.get("added_at")
        out.append(Record(
            id=r["id"], vector=vectors[i], label=r.get("label"), donor=r.get("donor"),
            counts=None if counts is None else np.asarray(counts, dtype=np.float32),
            added_at=None if added is None else datetime.fromisoformat(added),
            text=r.get("text"), payload=r.get("payload"),
            features=r.get("features", {}), source=r.get("source"),
        ))
    return out


def write_scimilarity_corpus(
    root: Path, n: int = 240, dim: int = 12, seed: int = 3,
    duplicate_ids: int = 0, truncate_meta: int = 0,
) -> Path:
    """A directory shaped like a released SCimilarity model.

    Small enough to run in a test, laid out exactly as the real one, so
    the adapter's sniffing, column detection and integrity checks are
    exercised against the genuine structure rather than a mock.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    cs = root / "cellsearch"
    cs.mkdir(parents=True, exist_ok=True)
    types = ["T cell", "B cell", "monocyte"]
    centres = rng.normal(size=(len(types), dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    vecs, rows = [], []
    for i in range(n):
        t = i % len(types)
        v = centres[t] + 0.25 * rng.normal(size=dim)
        v /= np.linalg.norm(v)
        vecs.append(v)
        rows.append({
            "index": f"cell{i:05d}",
            "celltype_name": types[t],
            "donor_id": f"donor{i % 5}",
            "study": f"study{t}",
        })
    for j in range(duplicate_ids):
        rows[j]["index"] = rows[0]["index"]

    np.save(cs / "full_kNN_embedding.npy", np.vstack(vecs).astype(np.float32))
    meta = pd.DataFrame(rows)
    if truncate_meta:
        meta = meta.iloc[:-truncate_meta]
    meta.to_csv(cs / "full_kNN_meta.csv", index=False)
    return root


def write_prodva_corpus(
    root: Path, n: int = 30, collide: int = 0, missing_from_map: int = 0,
    orphan_fragments: int = 0, seed: int = 5,
) -> Path:
    """A directory shaped like ProDVa's three attacker-writable files."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    families = ["kinase", "transporter", "protease", "binding protein"]
    data, seq_map, frag_map = [], [], []
    for i in range(n):
        fam = families[i % len(families)]
        desc = (f"The protein is a {fam} involved in cellular regulation, "
                f"catalysing step {i} of its pathway.")
        seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=60))
        data.append({"id": f"rec{i:04d}", "instruction": desc, "sequence": seq})
        seq_map.append({"instruction": desc, "sequence": seq})
        frag_map.append({"sequence": seq, "phrases": [
            {"phrase": seq[:20], "type": "domain", "name": f"{fam} domain",
             "description": f"{fam} catalytic region"},
        ]})
    for j in range(collide):
        data[j + 1]["instruction"] = data[0]["instruction"]
    for _ in range(missing_from_map):
        seq_map.pop()
    for j in range(orphan_fragments):
        frag_map.append({"sequence": f"ORPHANSEQ{j}", "phrases": [
            {"phrase": "QQQ", "type": "domain", "name": "orphan"}]})

    _dump(root / "data.jsonl", data)
    _dump(root / "sequence_map.jsonl", seq_map)
    _dump(root / "fragment_map.jsonl", frag_map)
    return root


def _dump(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
