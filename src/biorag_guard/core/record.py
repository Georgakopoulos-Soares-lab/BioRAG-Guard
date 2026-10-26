"""The vocabulary of the system: one word per concept, used everywhere.

The terms here are binding. ``access``, ``layer``, ``budget``, ``seasoned``,
``abstain`` and the verdict names appear identically in the CLI, the config
file, the report and the docs. Where a term already had a name in the
research code it keeps that name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .arrays import Array


class Access(str, Enum):
    """Which threat model a layer is evidence against.

    ``INSERT`` is the research's ``no_access`` arm: the attacker cannot write
    to the reference and can only get records accepted into it. ``OVERWRITE``
    is ``with_access``: the attacker edits records that are already there.
    The distinction decides whether a layer can run at ingest at all.
    """

    INSERT = "insert"
    OVERWRITE = "overwrite"


class Scope(str, Enum):
    """What a layer needs in order to run."""

    RECORD_LOCAL = "record-local"
    CORPUS_RELATIVE = "corpus-relative"
    CORPUS_WIDE = "corpus-wide"
    VOTE_TIME = "vote-time"


class Verdict(str, Enum):
    """The decision for one record. Ordered: admit < flag < quarantine < reject."""

    ADMIT = "admit"
    FLAG = "flag"
    QUARANTINE = "quarantine"
    REJECT = "reject"

    @property
    def rank(self) -> int:
        return _VERDICT_RANK[self]

    def worst(self, other: Verdict) -> Verdict:
        return self if self.rank >= other.rank else other


_VERDICT_RANK: dict[Verdict, int] = {
    Verdict.ADMIT: 0,
    Verdict.FLAG: 1,
    Verdict.QUARANTINE: 2,
    Verdict.REJECT: 3,
}

#: Process exit codes. 0 nothing above flag, 1 a quarantine, 2 a reject, 3 error.
EXIT_CODE: dict[Verdict, int] = {
    Verdict.ADMIT: 0,
    Verdict.FLAG: 0,
    Verdict.QUARANTINE: 1,
    Verdict.REJECT: 2,
}
EXIT_RUN_ERROR = 3


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Record:
    """One retrievable unit of a corpus: a cell, or a description-sequence pair.

    Attributes:
        id: stable identifier within the corpus.
        vector: the embedding actually stored in the index, or None if the
            adapter must produce it.
        label: the label the record CLAIMS. The word is deliberate: in the
            realistic attack the vector is genuine and the claim is not.
        text: a natural-language description (text-protein corpora).
        payload: the thing the description is bonded to (a protein sequence).
        counts: raw, pre-normalisation measurements, if the corpus carries
            them. ``recompute`` is unavailable without this.
        donor: provenance group used by ``donor_cap``.
        added_at: when the record entered the corpus; drives seasoning.
        features: adapter-computed quantities (QC statistics, fragment lists).
        metadata: everything else the corpus knows about the record.
        source: the file, study or batch it arrived in.
    """

    id: str
    vector: Array | None = None
    label: str | None = None
    text: str | None = None
    payload: str | None = None
    counts: Any | None = None
    donor: str | None = None
    added_at: datetime | None = None
    features: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    def capabilities(self) -> set[str]:
        """Which evidence this record actually carries."""
        caps: set[str] = set()
        if self.vector is not None:
            caps.add("vector")
        if self.label is not None:
            caps.add("label")
        if self.text is not None:
            caps.add("text")
        if self.payload is not None:
            caps.add("payload")
        if self.counts is not None:
            caps.add("counts")
        if self.donor is not None:
            caps.add("donor")
        if self.added_at is not None:
            caps.add("added_at")
        return caps
