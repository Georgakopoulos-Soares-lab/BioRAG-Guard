"""Cross-modal layers: does the description actually describe the payload?

This is where the protein-arm evidence points. The random control -- the
same payload attached to an unrelated description -- captured 0 of 100
queries, while the targeted version captured up to 0.97. The payload is
not what wins retrieval; the *bond* between a plausible description and an
unrelated payload is the attack. Verifying that bond is therefore the
highest-value check available on a text-keyed corpus, and the only one
with any purchase on the quiet, genuinely-paraphrased arm.

The agreement backend is pluggable and **no default scores biology**.
``TermOverlapBackend`` compares the description against annotation terms
the adapter supplies with the record; if nothing supplies them, the layer
reports itself as not applicable rather than inventing a score. Wiring a
real annotation source (profile-HMM search, InterProScan, or a
text-protein embedding model) is a deployment decision -- see
``docs/decisions/0002-agreement-backend.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core.arrays import Array
from ..core.finding import Finding
from ..core.record import Access, Record, Scope, Severity, Verdict
from ..core.registry import Context, Direction, Layer, LayerSpec, register_layer
from .text import tokens


@runtime_checkable
class AgreementBackend(Protocol):
    """Scores how well a description matches what its payload actually is."""

    name: str

    def available(self, records: Sequence[Record]) -> bool:
        """Can this backend score these records at all?"""

    def score(self, records: Sequence[Record]) -> Array:
        """(n,) agreement in [0, 1]; higher means description and payload agree."""


class TermOverlapBackend:
    """Compare description wording with annotation terms carried by the record.

    The annotation terms must come from somewhere real: the adapter reads
    them from the corpus's own fragment or domain annotations and puts
    them in ``record.features['annotation_terms']``. This backend does not
    derive biology from the sequence; it checks that two things the corpus
    already asserts are consistent with each other.
    """

    name = "term_overlap"

    def available(self, records: Sequence[Record]) -> bool:
        return all(
            r.features and r.features.get("annotation_terms") for r in records
        )

    def score(self, records: Sequence[Record]) -> Array:
        out = np.ones(len(records))
        for i, rec in enumerate(records):
            terms = {t.lower() for t in rec.features.get("annotation_terms", [])}
            words = set(tokens(rec.text or ""))
            if not terms:
                out[i] = 1.0  # no evidence is not evidence against
                continue
            hit = sum(1 for t in terms if any(w in t.split() or t in w for w in words))
            out[i] = hit / len(terms)
        return out


#: Backends by name. A deployment registers its own here.
BACKENDS: dict[str, AgreementBackend] = {"term_overlap": TermOverlapBackend()}


@register_layer
class CrossModalAgreement(Layer):
    """Does the description match what the payload is annotated to be?

    Targets the attack's defining shape directly: a real-looking
    description welded to an unrelated payload. Unlike near-duplicate
    detection it does not depend on the poison being a copy, which is why
    it is the layer with any claim on the paraphrased arm.
    """

    spec = LayerSpec(
        name="crossmodal_agreement",
        scope=Scope.RECORD_LOCAL,
        access=frozenset({Access.INSERT, Access.OVERWRITE}),
        needs=frozenset({"text", "payload"}),
        catches="A plausible description bonded to a payload that is annotated as "
                "something else -- the shape the random-control arm isolates.",
        misses="A payload whose annotation genuinely matches the description, and any "
               "record whose annotation terms the corpus does not carry.",
        statistic="agreement between the description and the payload's annotation terms, "
                  "via the configured backend",
        direction=Direction.LOW_IS_SUSPICIOUS,
        default_verdict=Verdict.QUARANTINE,
        severity=Severity.HIGH,
        default_params={"backend": "term_overlap"},
    )

    def _backend(self) -> AgreementBackend:
        name = str(self.params["backend"])
        try:
            return BACKENDS[name]
        except KeyError:
            raise NotImplementedError(
                f"no agreement backend named {name!r}; registered: {sorted(BACKENDS)}"
            ) from None

    def score(self, records: Sequence[Record], ctx: Context) -> Array:
        backend = self._backend()
        if not backend.available(records):
            raise NotImplementedError(
                f"the {backend.name!r} agreement backend has nothing to score: these "
                "records carry no annotation terms. Configure an annotation source, or "
                "disable crossmodal_agreement rather than reading its silence as a pass."
            )
        return backend.score(records)

    def evidence(self, record: Record, index: int, ctx: Context) -> dict[str, Any]:
        return {
            "backend": str(self.params["backend"]),
            "annotation_terms": list(record.features.get("annotation_terms", []))[:10],
        }

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        terms = ", ".join(evidence.get("annotation_terms", [])[:3])
        return (
            f"The description does not match what this payload is annotated to be "
            f"(agreement {score:.2f}; annotated as {terms}). A real description attached "
            f"to an unrelated sequence is what makes a poisoned record retrievable."
        )

    def remedy(self, record: Record) -> str:
        return ("Confirm the description and the sequence came from the same source "
                "record before accepting.")


@register_layer
class CrossModalMaps(Layer):
    """Consistency across the artifacts a text-protein corpus is made of.

    ProDVa's corpus is three files an attacker can write: the index built
    from descriptions, the description to sequence map, and the sequence
    to fragments map. Upstream collapses colliding description keys
    silently, and a payload missing from the fragment map contributes
    nothing. Set algebra over the three is deterministic and cheap, so it
    runs by default.
    """

    spec = LayerSpec(
        name="crossmodal_maps",
        scope=Scope.CORPUS_WIDE,
        access=frozenset({Access.INSERT, Access.OVERWRITE}),
        needs=frozenset(),
        catches="Descriptions indexed but absent from the sequence map, sequences absent "
                "from the fragment map, orphaned fragment entries, and colliding "
                "description keys that silently overwrite a real record.",
        misses="Poison that is internally consistent across all three files.",
        statistic="set algebra over the index, the sequence map and the fragment map",
        budgeted=False,
        default_verdict=Verdict.REJECT,
        severity=Severity.HIGH,
    )

    def run(self, records: Sequence[Record], ctx: Context) -> list[Finding]:
        adapter = ctx.adapter
        if adapter is None or not hasattr(adapter, "map_integrity"):
            raise NotImplementedError(
                "this corpus is not made of description/sequence/fragment maps"
            )
        return list(adapter.map_integrity())
