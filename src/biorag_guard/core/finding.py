"""A finding, and the report that collects them.

Two rules are enforced by the types rather than by convention.

**Retrieval failure and outcome failure are separate fields.** Every layer
ported here is a vote-time filter; in the research grid not one of them
changed the retrieval rate at all. Collapsing the two into a single
"did the defence work" number would read as total failure.

**A layer that did not run is recorded.** ``Report.skipped`` carries the
layers that were switched off or unusable and why, so silence in the
findings list never has to be interpreted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .record import EXIT_CODE, Severity, Verdict

#: Bumped whenever the JSON report changes shape.
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Finding:
    """One layer firing on one record."""

    layer: str
    record_id: str
    score: float
    threshold: float
    severity: Severity
    verdict: Verdict
    reason: str
    remedy: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "record_id": self.record_id,
            "score": self.score,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "remedy": self.remedy,
            "evidence": _jsonable(self.evidence),
        }


@dataclass(frozen=True)
class Skipped:
    """A layer that did not run, and the plain reason why."""

    layer: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"layer": self.layer, "reason": self.reason}


@dataclass(frozen=True)
class RecordResult:
    """The verdict for one record and the findings behind it."""

    record_id: str
    verdict: Verdict
    findings: tuple[Finding, ...] = ()

    @property
    def reasons(self) -> list[str]:
        return [f.reason for f in self.findings]


@dataclass
class Report:
    """Everything one run produced.

    Attributes:
        command: which job ran -- ``check``, ``scan``, ``calibrate``.
        corpus: human-readable corpus path.
        adapter: adapter name.
        results: per-record verdicts (``check``) -- may be empty for a sweep.
        findings: every finding, including corpus-wide ones with no record.
        skipped: layers that did not run, with the reason.
        retrieval: corpus-wide retrieval-side measurements.
        outcome: corpus-wide outcome-side measurements.
        calibration: per-layer threshold, budget, and the cost it bought.
        provenance: versions, seeds, sample sizes, timings.
        notes: sentences the user must see, e.g. that sampling occurred.
    """

    command: str
    corpus: str
    adapter: str
    results: list[RecordResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    retrieval: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def verdict(self) -> Verdict:
        """The worst verdict anywhere in the run."""
        worst = Verdict.ADMIT
        for r in self.results:
            worst = worst.worst(r.verdict)
        for f in self.findings:
            worst = worst.worst(f.verdict)
        return worst

    @property
    def exit_code(self) -> int:
        return EXIT_CODE[self.verdict]

    def counts(self) -> dict[str, int]:
        """How many records landed on each verdict."""
        out = {v.value: 0 for v in Verdict}
        for r in self.results:
            out[r.verdict.value] += 1
        return out

    def by_layer(self) -> dict[str, int]:
        """How many findings each layer produced, worst-first by verdict."""
        tally: dict[str, int] = {}
        for f in self.findings:
            tally[f.layer] = tally.get(f.layer, 0) + 1
        return dict(sorted(tally.items(), key=lambda kv: -kv[1]))

    def add(self, findings: Iterable[Finding]) -> None:
        self.findings.extend(findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "command": self.command,
            "created_at": self.created_at,
            "corpus": self.corpus,
            "adapter": self.adapter,
            "verdict": self.verdict.value,
            "exit_code": self.exit_code,
            "counts": self.counts(),
            "notes": list(self.notes),
            "retrieval": _jsonable(self.retrieval),
            "outcome": _jsonable(self.outcome),
            "calibration": _jsonable(self.calibration),
            "findings": [f.to_dict() for f in self.findings],
            "skipped": [s.to_dict() for s in self.skipped],
            "results": [
                {"record_id": r.record_id, "verdict": r.verdict.value,
                 "layers": [f.layer for f in r.findings]}
                for r in self.results
            ],
            "provenance": _jsonable(self.provenance),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


def _jsonable(obj: Any) -> Any:
    """Make numpy and set types survive ``json.dumps`` without surprises."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_jsonable(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (Severity, Verdict)):
        return obj.value
    return obj
