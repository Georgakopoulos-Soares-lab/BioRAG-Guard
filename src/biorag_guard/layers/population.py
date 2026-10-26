"""Corpus-wide layers: what a per-record check structurally cannot see.

A hub record looks unremarkable on its own and is defined entirely by how
often the corpus returns it. A duplicate cluster is invisible until you
look at the population. These need the sweep, which is why ``bioguard
check`` tells the user to schedule a ``bioguard scan`` rather than implying
ingest screening is complete coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.finding import Finding
from ..core.record import Access, Record, Scope, Severity, Verdict
from ..core.registry import Context, Direction, Layer, LayerSpec, register_layer


def k_occurrence(adapter: Any, n_probes: int, k: int, seed: int) -> np.ndarray:
    """How often each corpus record is returned for a probe query.

    Hubness is the attack in a similarity-search deployment: being returned
    IS the exploit, with no generation step required.

    Returns:
        (n_records,) int counts.
    """
    probes = adapter.probe_queries(n_probes, seed=seed)
    idx, _ = adapter.neighbors(probes, k)
    counts = np.zeros(len(adapter.ids), dtype=np.int64)
    flat = np.asarray(idx).ravel()
    flat = flat[flat >= 0]
    np.add.at(counts, flat, 1)
    return counts


@register_layer
class Hub(Layer):
    """A record returned for a large share of unrelated queries.

    Universal hubs are invisible to any per-record check: nothing about the
    record itself is wrong, and the property only exists relative to the
    query distribution. In the protein arm, 1,500 hub records reached 0.60
    retrieval capture against free-text requests.
    """

    spec = LayerSpec(
        name="hub",
        scope=Scope.CORPUS_WIDE,
        access=frozenset({Access.INSERT, Access.OVERWRITE}),
        needs=frozenset({"vector"}),
        catches="Records positioned to be retrieved for many unrelated queries at once.",
        misses="Targeted poison aimed at a single query, which by construction is returned "
               "rarely. Not detectable at ingest at all -- only a sweep can see it.",
        statistic="k-occurrence: how many of the probe queries return this record",
        direction=Direction.HIGH_IS_SUSPICIOUS,
        default_verdict=Verdict.QUARANTINE,
        severity=Severity.HIGH,
        default_params={"probes": 5000, "k": 50, "seed": 0},
    )

    def _counts(self, ctx: Context) -> np.ndarray:
        key = f"k_occurrence:{self.params['probes']}:{self.params['k']}:{self.params['seed']}"
        if key not in ctx.cache:
            ctx.cache[key] = k_occurrence(
                ctx.adapter, int(self.params["probes"]), int(self.params["k"]),
                int(self.params["seed"]),
            )
        return ctx.cache[key]  # type: ignore[no-any-return]

    def score(self, records: Sequence[Record], ctx: Context) -> np.ndarray:
        if ctx.adapter is None:
            raise NotImplementedError("hub needs an adapter to probe")
        counts = self._counts(ctx)
        lookup = {str(v): i for i, v in enumerate(np.asarray(ctx.adapter.ids, dtype=object))}
        return np.array(
            [float(counts[lookup[r.id]]) if r.id in lookup else 0.0 for r in records]
        )

    def evidence(self, record: Record, index: int, ctx: Context) -> dict[str, Any]:
        return {"probes": int(self.params["probes"]), "k": int(self.params["k"])}

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        probes = evidence.get("probes", self.params["probes"])
        pct = 100.0 * score / max(probes, 1)
        return (
            f"This record is returned for {int(score):,} of {probes:,} probe queries "
            f"({pct:.1f}%). A record that answers everything answers nothing honestly."
        )

    def remedy(self, record: Record) -> str:
        return ("Hold the record out of the index and re-run the sweep; if capture drops, "
                "the record was acting as a hub.")


@register_layer
class DuplicateCluster(Layer):
    """Groups of near-identical records inside the corpus.

    A budgeted attacker inserts many near-copies rather than one, so the
    population signature is a tight cluster. Reported per cluster, not per
    record, because the cluster is the thing a user acts on.
    """

    spec = LayerSpec(
        name="duplicate_cluster",
        scope=Scope.CORPUS_WIDE,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"vector"}),
        catches="Many near-identical records already sitting in the corpus, including ones "
                "that predate any ingest screening.",
        misses="A single copy, which `dedup` catches at ingest, and distinct synthesised "
               "records, which form no cluster.",
        statistic="connected components of the cosine>tau graph, reported at size>=min_size",
        budgeted=False,
        default_verdict=Verdict.FLAG,
        severity=Severity.MEDIUM,
        default_params={"tau": 0.999, "min_size": 5, "block": 2048},
    )

    def run(self, records: Sequence[Record], ctx: Context) -> list[Finding]:
        adapter = ctx.adapter
        if adapter is None or adapter.vectors is None:
            raise NotImplementedError("duplicate_cluster needs a corpus with vectors")
        tau = float(self.params["tau"])
        min_size = int(self.params["min_size"])
        vecs = np.asarray(adapter.vectors, dtype=np.float32)
        ids = np.asarray(adapter.ids, dtype=object)
        parent = list(range(len(vecs)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        block = int(self.params["block"])
        for lo in range(0, len(vecs), block):
            hi = min(lo + block, len(vecs))
            sim = vecs[lo:hi] @ vecs.T
            rows, cols = np.nonzero(sim >= tau)
            for r, c in zip(rows.tolist(), cols.tolist()):
                if lo + r != c:
                    union(lo + r, c)

        groups: dict[int, list[int]] = {}
        for i in range(len(vecs)):
            groups.setdefault(find(i), []).append(i)

        out: list[Finding] = []
        for root, members in sorted(groups.items()):
            if len(members) < min_size:
                continue
            out.append(
                Finding(
                    layer=self.name,
                    record_id=str(ids[root]),
                    score=float(len(members)),
                    threshold=float(min_size),
                    severity=self.spec.severity,
                    verdict=self.verdict,
                    reason=(
                        f"{len(members)} records in this corpus are near-identical to one "
                        f"another (cosine >= {tau:g}). Together they carry "
                        f"{len(members)} votes for what is effectively one measurement."
                    ),
                    remedy="Check whether the group came from one submission; collapse it "
                           "to a single record or cap its votes per donor.",
                    evidence={
                        "size": len(members),
                        "tau": tau,
                        "member_ids": [str(ids[m]) for m in members[:50]],
                        "truncated": len(members) > 50,
                    },
                )
            )
        return out


@register_layer
class IdxMeta(Layer):
    """Structural consistency between the index and the metadata beside it.

    SCimilarity's own build notes say the metadata process is not hardened
    and the index column is not guaranteed unique; a repeated key means one
    record silently overwrites another. This layer asks the adapter, which
    is the only component that knows the corpus layout.
    """

    spec = LayerSpec(
        name="idx_meta",
        scope=Scope.CORPUS_WIDE,
        access=frozenset({Access.INSERT, Access.OVERWRITE}),
        needs=frozenset(),
        catches="Index/metadata row-count mismatch, duplicate primary keys, orphaned "
                "sidecar files.",
        misses="Metadata that is internally consistent and wrong.",
        statistic="set algebra and row counts across the corpus files",
        budgeted=False,
        default_verdict=Verdict.REJECT,
        severity=Severity.HIGH,
    )

    def run(self, records: Sequence[Record], ctx: Context) -> list[Finding]:
        if ctx.adapter is None:
            raise NotImplementedError("idx_meta needs an adapter")
        return list(ctx.adapter.integrity())
