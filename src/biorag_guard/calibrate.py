"""Fit thresholds to a budget, and measure what they cost on clean data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core.baseline import Baseline, fit_threshold
from .core.record import Record
from .core.registry import Context, Layer
from .vote import VotePolicy, defended_vote


@dataclass
class Split:
    """A fit/held-out split of records believed genuine."""

    fit: list[Record]
    held_out: list[Record]

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.fit), len(self.held_out)


def sample_split(
    adapter: Any, n: int, seed: int = 0, held_out_fraction: float = 0.2
) -> Split:
    """Sample the corpus and split it. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    total = len(adapter)
    take = min(n, total)
    pick = np.sort(rng.choice(total, size=take, replace=False))
    records = list(adapter.iter_records())
    chosen = [records[i] for i in pick]
    rng.shuffle(chosen)
    cut = max(1, int(round(len(chosen) * (1.0 - held_out_fraction))))
    return Split(fit=chosen[:cut], held_out=chosen[cut:] or chosen[-1:])


def calibrate_layers(
    layers: Sequence[Layer],
    split: Split,
    ctx: Context,
    budget: float,
    baseline: Baseline,
) -> tuple[Baseline, list[tuple[str, str]]]:
    """Fit every budgeted layer. Returns the baseline and what was skipped."""
    skipped: list[tuple[str, str]] = []
    for layer in layers:
        if not layer.spec.budgeted:
            continue
        try:
            scores = np.asarray(layer.score(split.held_out, ctx), dtype=float)
            baseline.layers[layer.name] = fit_threshold(
                layer.name, scores, layer.spec.direction, budget
            )
        except (NotImplementedError, ValueError) as exc:
            skipped.append((layer.name, str(exc)))
    return baseline, skipped


def clean_cost(
    adapter: Any,
    split: Split,
    policy: VotePolicy,
    k: int = 50,
    lc_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """What the vote policy costs on clean data.

    Measured the way the research measures it: the same held-out records,
    voted twice -- once with every layer off, once with the policy on --
    so the difference carries no sampling variance. Abstentions are
    reported separately and are **not** counted as errors.
    """
    if adapter.vectors is None or adapter.labels is None:
        return {"available": False,
                "why": "this corpus has no labels, so there is no vote to cost"}
    ids = np.asarray(adapter.ids, dtype=object)
    labels = np.asarray(adapter.labels, dtype=object)
    donors = None if adapter.donors is None else np.asarray(adapter.donors, dtype=object)
    vectors = np.asarray(adapter.vectors)
    lookup = {str(v): i for i, v in enumerate(ids)}

    probes = [r for r in split.held_out if r.vector is not None and r.label is not None]
    if not probes:
        return {"available": False, "why": "no held-out records carried a vector and label"}

    q = np.vstack([np.asarray(r.vector, dtype=np.float32) for r in probes])
    idx, cos = adapter.neighbors(q, k + 1)

    base_right = defended_right = abstained = 0
    for row, rec in enumerate(probes):
        self_i = lookup.get(rec.id, -1)
        keep = [j for j in range(idx.shape[1]) if idx[row, j] != self_i][:k]
        nbr = idx[row, keep]
        nbr_cos = cos[row, keep]
        nbr_lab = labels[nbr]
        undefended = defended_vote(nbr_lab, VotePolicy(), cos=nbr_cos)
        lc = None
        if policy.label_consistency and lc_scores is not None:
            lc = np.array([lc_scores.get(str(ids[j]), 1.0) for j in nbr])
        out = defended_vote(
            nbr_lab, policy,
            recompute_ok=np.ones(len(nbr), dtype=bool) if policy.recompute else None,
            label_consistency=lc,
            donor=None if donors is None else donors[nbr],
            seasoned=np.ones(len(nbr), dtype=bool) if policy.quarantine else None,
            cos=nbr_cos, emb=vectors[nbr] if policy.dedup_tau else None,
        )
        base_right += int(undefended == rec.label)
        if out is None:
            abstained += 1
        else:
            defended_right += int(out == rec.label)

    n = len(probes)
    return {
        "available": True,
        "n_probes": n,
        "k": k,
        "policy": policy.name,
        "clean_accuracy": base_right / n,
        "clean_accuracy_defended": defended_right / n,
        "accuracy_delta": (defended_right - base_right) / n,
        "abstention_rate": abstained / n,
    }
