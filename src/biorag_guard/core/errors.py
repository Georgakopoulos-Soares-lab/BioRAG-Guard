"""Errors that say what to do next.

``MissingEvidence`` is the important one. The research code raises rather
than defaulting to pass when a layer is enabled without the evidence it
needs, because reporting a defence that never ran as a defence that found
nothing is a false negative dressed as a clean result. That behaviour is
preserved here.
"""

from __future__ import annotations


class BioGuardError(Exception):
    """Base class for every error this package raises on purpose."""


class MissingEvidence(BioGuardError):
    """A layer is enabled but the evidence it needs was not supplied."""


class AdapterNotFound(BioGuardError):
    """No adapter recognised the corpus."""


class ExtraNotInstalled(BioGuardError):
    """An adapter needs an optional dependency that is not installed."""

    def __init__(self, adapter: str, extra: str, package: str) -> None:
        super().__init__(
            f"the {adapter!r} adapter needs {package!r}, which is not installed.\n"
            f"  pip install 'biorag-guard[{extra}]'"
        )


class BaselineMismatch(BioGuardError):
    """The stored baseline was fitted on a different corpus."""


class CorpusAlreadyContainsBatch(BioGuardError):
    """A batch was submitted for ingest screening after it was already ingested.

    ``label_consistency`` scores a record against the reference as it stood
    BEFORE the batch. If the batch is already in the index, copies of one
    poisoned record corroborate each other and the layer silently weakens.
    """


class Quarantined(BioGuardError):
    """Raised by ``@guard.gate`` when a record does not pass."""

    def __init__(self, record_id: str, verdict: str, reasons: list[str]) -> None:
        self.record_id = record_id
        self.verdict = verdict
        self.reasons = reasons
        super().__init__(f"record {record_id!r}: {verdict} - " + "; ".join(reasons))
