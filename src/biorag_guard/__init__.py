"""BioRAG-Guard: integrity checks for biological retrieval corpora.

Two modes, matching the two threat models the research separates:

* **check at ingest** (``Guard.check``) -- the attacker can only get records
  accepted into the corpus;
* **scan a stored corpus** (``Guard.scan``) -- the attacker can edit records that
  are already there.

A third surface, ``Guard.vote``, applies the query-time policy, because
``donor_cap``, ``weight`` and ``abstain`` have no ingest-time form.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core.errors import (
    AdapterNotFound,
    BaselineMismatch,
    BioGuardError,
    CorpusAlreadyContainsBatch,
    ExtraNotInstalled,
    MissingEvidence,
    Quarantined,
)
from .core.finding import Finding, RecordResult, Report, Skipped
from .core.record import Access, Record, Scope, Severity, Verdict
from .guard import Guard
from .vote import VotePolicy

__all__ = [
    "__version__",
    "Access", "AdapterNotFound", "BaselineMismatch", "BioGuardError",
    "CorpusAlreadyContainsBatch", "ExtraNotInstalled", "Finding", "Guard",
    "MissingEvidence", "Quarantined", "Record", "RecordResult", "Report",
    "Scope", "Severity", "Skipped", "Verdict", "VotePolicy",
]
