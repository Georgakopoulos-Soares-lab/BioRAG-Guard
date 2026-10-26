"""Budget-calibrated thresholds, and the cost they buy.

Every threshold in this package is derived from a stated budget -- the
fraction of genuine records a layer is allowed to sacrifice -- measured on
a held-out split of the corpus. No constant from a paper, and no 0.5.

The cost block is not decoration. The research sets the label-consistency
threshold by budget precisely because 0.5 is arbitrary, and reports the
benefit and the genuine-record cost on the same row. A tool that cannot
say what its defence cost on clean data is not usable for the decision the
user actually faces.

One warning is mandatory and the CLI prints it: if the corpus is already
poisoned, the baseline absorbs the poison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BaselineMismatch
from .registry import Direction

BASELINE_VERSION = "1.0"


@dataclass
class LayerBaseline:
    """One layer's calibration."""

    layer: str
    threshold: float
    budget: float
    direction: str
    n_fitted: int
    flagged_fraction: float
    quantiles: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Baseline:
    """Thresholds plus the fingerprint of the corpus they were fitted on."""

    adapter: str
    corpus: str
    n_records: int
    dims: int | None
    budget: float
    created_at: str
    seed: int
    layers: dict[str, LayerBaseline] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    version: str = BASELINE_VERSION

    # -- fingerprint ------------------------------------------------------

    def fingerprint(self) -> tuple[str, int, int | None]:
        return (self.adapter, self.n_records, self.dims)

    def check_against(self, adapter: Any) -> None:
        """Refuse to run on a corpus the baseline was not fitted on."""
        dims = None if adapter.vectors is None else int(adapter.vectors.shape[1])
        now = (adapter.name, len(adapter), dims)
        if now != self.fingerprint():
            raise BaselineMismatch(
                f"this baseline was fitted on {self.fingerprint()} "
                f"(adapter, records, dims) but the corpus is {now}.\n"
                f"  re-run: bioguard calibrate {adapter.path}"
            )

    # -- threshold retuning -------------------------------------------------

    def retune(self, budget: float) -> Baseline:
        """New thresholds at a different budget, from the stored quantiles.

        No refitting: the quantile curve was stored at calibration time
        precisely so ``--budget`` is a dial rather than a 12-minute wait.
        """
        out = Baseline(
            adapter=self.adapter, corpus=self.corpus, n_records=self.n_records,
            dims=self.dims, budget=budget, created_at=self.created_at, seed=self.seed,
            cost=dict(self.cost),
            notes=[*self.notes, f"thresholds retuned from budget {self.budget} to {budget}"],
        )
        for name, lb in self.layers.items():
            q = _budget_quantile(Direction(lb.direction), budget)
            key = _qkey(q)
            if key not in lb.quantiles:
                out.layers[name] = lb
                out.notes.append(
                    f"{name}: no stored quantile at {key}; threshold left at {lb.threshold}"
                )
                continue
            out.layers[name] = LayerBaseline(
                layer=name, threshold=lb.quantiles[key], budget=budget,
                direction=lb.direction, n_fitted=lb.n_fitted,
                flagged_fraction=budget, quantiles=lb.quantiles,
            )
        return out

    # -- io -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "adapter": self.adapter,
            "corpus": self.corpus,
            "n_records": self.n_records,
            "dims": self.dims,
            "budget": self.budget,
            "created_at": self.created_at,
            "seed": self.seed,
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "cost": self.cost,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        raw = json.loads(Path(path).read_text())
        raw.pop("version", None)
        layers = {k: LayerBaseline(**v) for k, v in raw.pop("layers", {}).items()}
        return cls(layers=layers, **raw)

    @classmethod
    def empty(cls, adapter: Any, budget: float, seed: int = 0) -> Baseline:
        return cls(
            adapter=adapter.name,
            corpus=str(adapter.path),
            n_records=len(adapter),
            dims=None if adapter.vectors is None else int(adapter.vectors.shape[1]),
            budget=budget,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            seed=seed,
        )


#: Budgets the research used, offered as presets.
BUDGET_LADDER = (0.01, 0.05, 0.1)
#: Quantiles stored so ``--budget`` can retune without refitting.
STORED_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def _budget_quantile(direction: Direction, budget: float) -> float:
    """Which quantile of the genuine-score distribution the threshold sits at."""
    return budget if direction is Direction.LOW_IS_SUSPICIOUS else 1.0 - budget


def _qkey(q: float) -> str:
    return f"q{q:.4f}"


def fit_threshold(
    layer_name: str,
    scores: np.ndarray,
    direction: Direction,
    budget: float,
) -> LayerBaseline:
    """Set a threshold that sacrifices at most ``budget`` of these records.

    Args:
        scores: the layer's score on records believed genuine.
        direction: which end of the score is suspicious.
        budget: the fraction of genuine records the layer may flag.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError(f"{layer_name}: no finite scores to calibrate on")
    q = _budget_quantile(direction, budget)
    threshold = float(np.quantile(s, q))
    fired = (s < threshold) if direction is Direction.LOW_IS_SUSPICIOUS else (s > threshold)
    stored = {
        _qkey(_budget_quantile(direction, b)): float(
            np.quantile(s, _budget_quantile(direction, b))
        )
        for b in STORED_BUDGETS
    }
    return LayerBaseline(
        layer=layer_name,
        threshold=threshold,
        budget=budget,
        direction=direction.value,
        n_fitted=int(s.size),
        flagged_fraction=float(fired.mean()),
        quantiles=stored,
    )
