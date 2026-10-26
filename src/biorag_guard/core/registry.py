"""The layer registry: how a defence declares itself and what it needs.

A layer is one defence, switchable alone. That is a product requirement
rather than a research nicety -- the measured grid contains at least one
configuration where a layer is actively harmful, and a user who cannot see
what a layer cost them cannot decide whether to keep it.

Most layers are **budgeted**: they reduce a record to one score, and their
threshold is whatever sacrifices at most ``budget`` of genuine records. Such
a layer implements :meth:`Layer.score` and gets its threshold, its reason
text and its calibration for free. A layer that is not a scorer -- an
integrity check, a set-algebra check -- overrides :meth:`Layer.run`.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .errors import MissingEvidence
from .finding import Finding
from .record import Access, Record, Scope, Severity, Verdict


class Direction(str, Enum):
    """Which end of the score is suspicious. Decides the budget quantile."""

    LOW_IS_SUSPICIOUS = "low"
    HIGH_IS_SUSPICIOUS = "high"


@dataclass(frozen=True)
class LayerSpec:
    """Everything ``bioguard explain`` prints, written as data not prose.

    Attributes:
        name: the ubiquitous-language name. Ported layers keep the research
            name exactly.
        scope: what the layer needs in order to run.
        access: which threat models it is evidence against.
        needs: record capabilities required, e.g. ``{"vector", "label"}``.
        catches: what it is measured to stop.
        misses: what it provably does NOT stop. Never left empty.
        statistic: the quantity it computes, in one line.
        ported: True if the implementation comes from the research code and
            must keep its semantics.
        budgeted: True if its threshold is set by a budget.
        direction: which end of the score is suspicious.
        default_verdict: the verdict a firing produces unless overridden.
        severity: reporting severity.
        ingest_only: the layer is unsound at query time (``dedup``).
    """

    name: str
    scope: Scope
    access: frozenset[Access]
    needs: frozenset[str]
    catches: str
    misses: str
    statistic: str
    ported: bool = False
    budgeted: bool = True
    direction: Direction = Direction.HIGH_IS_SUSPICIOUS
    default_verdict: Verdict = Verdict.FLAG
    severity: Severity = Severity.MEDIUM
    ingest_only: bool = False
    default_params: dict[str, Any] = field(default_factory=dict)


class Context:
    """What a layer may look at besides the records it was handed.

    Attributes:
        adapter: the open corpus. ``None`` for pure record-local layers in
            unit tests.
        thresholds: calibrated thresholds by layer name.
        params: per-layer parameters from the config.
        rng: the single seeded generator; layers never make their own.
        batch_ids: ids of the batch under ingest screening, so a layer can
            refuse to score a batch against a reference already holding it.
    """

    def __init__(
        self,
        adapter: Any = None,
        thresholds: dict[str, float] | None = None,
        params: dict[str, dict[str, Any]] | None = None,
        rng: np.random.Generator | None = None,
        batch_ids: Sequence[str] | None = None,
    ) -> None:
        self.adapter = adapter
        self.thresholds = thresholds or {}
        self.params = params or {}
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.batch_ids = list(batch_ids) if batch_ids is not None else None
        #: Scratch shared between layers within one run -- a probe sweep or a
        #: manifest diff that several layers read. Nothing here survives the run.
        self.cache: dict[str, Any] = {}


class Layer(ABC):
    """One defence."""

    spec: LayerSpec

    def __init__(self, threshold: float | None = None, verdict: Verdict | None = None,
                 **params: Any) -> None:
        self.params: dict[str, Any] = {**self.spec.default_params, **params}
        self._threshold = threshold
        self.verdict = verdict or self.spec.default_verdict

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise MissingEvidence(
                f"layer {self.name!r} is budgeted but has no threshold; "
                "run `bioguard calibrate` first"
            )
        return self._threshold

    def set_threshold(self, value: float) -> None:
        self._threshold = value

    # -- the two things a layer may implement ---------------------------

    def score(self, records: Sequence[Record], ctx: Context) -> np.ndarray:
        """One score per record. Budgeted layers implement this."""
        raise NotImplementedError

    def fires(self, scores: np.ndarray) -> np.ndarray:
        """Which scores are on the suspicious side of the threshold."""
        if self.spec.direction is Direction.LOW_IS_SUSPICIOUS:
            return scores < self.threshold
        return scores > self.threshold

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        """One plain sentence for a bench scientist. Overridden per layer."""
        return f"{self.name}: score {score:.4g} against threshold {self.threshold:.4g}"

    def remedy(self, record: Record) -> str:
        return "Review this record before accepting it into the corpus."

    def evidence(self, record: Record, index: int, ctx: Context) -> dict[str, Any]:
        return {}

    def check_evidence(self, records: Sequence[Record]) -> None:
        """Raise if the records lack what this layer needs.

        Defaulting a missing score to "passed" would report a layer that
        never ran as a layer that found nothing.
        """
        if not records:
            return
        have = set.intersection(*(r.capabilities() for r in records)) if records else set()
        missing = set(self.spec.needs) - have
        if missing:
            raise MissingEvidence(
                f"layer {self.name!r} needs {sorted(missing)} but the records do not "
                f"carry it; disable the layer or supply the evidence"
            )

    def run(self, records: Sequence[Record], ctx: Context) -> list[Finding]:
        """Findings for these records. Scorer layers get this for free."""
        self.check_evidence(records)
        if not records:
            return []
        scores = np.asarray(self.score(records, ctx), dtype=float)
        if scores.shape != (len(records),):
            raise ValueError(
                f"{self.name}.score returned {scores.shape}, expected {(len(records),)}"
            )
        out: list[Finding] = []
        for i, fired in enumerate(self.fires(scores)):
            if not fired:
                continue
            rec = records[i]
            ev = self.evidence(rec, i, ctx)
            out.append(
                Finding(
                    layer=self.name,
                    record_id=rec.id,
                    score=float(scores[i]),
                    threshold=float(self.threshold),
                    severity=self.spec.severity,
                    verdict=self.verdict,
                    reason=self.reason(rec, float(scores[i]), ev),
                    remedy=self.remedy(rec),
                    evidence=ev,
                )
            )
        return out


_REGISTRY: dict[str, type[Layer]] = {}


def register_layer(cls: type[Layer]) -> type[Layer]:
    """Class decorator: make a layer discoverable by name."""
    name = cls.spec.name
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"layer name {name!r} is already registered")
    _REGISTRY[name] = cls
    return cls


def get_layer(name: str) -> type[Layer]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown layer {name!r}; known layers: {sorted(_REGISTRY)}"
        ) from None


def all_layers() -> dict[str, type[Layer]]:
    """Every registered layer, name -> class."""
    return dict(sorted(_REGISTRY.items()))


def layers_for(access: Access | None = None, scope: Scope | None = None) -> list[type[Layer]]:
    out = []
    for cls in all_layers().values():
        if access is not None and access not in cls.spec.access:
            continue
        if scope is not None and cls.spec.scope is not scope:
            continue
        out.append(cls)
    return out
