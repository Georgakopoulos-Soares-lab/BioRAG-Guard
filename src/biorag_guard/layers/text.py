"""Layers for text-keyed corpora, such as ProDVa's description index.

What the protein-arm measurements say about where to spend effort: a
record whose description is a near-verbatim copy of an expected request
captures 0.58 of the time against templated queries and 0.97 against
free-text ones, while a genuinely paraphrased one captures 0.21. So
near-duplicate detection is necessary, cheap, and only covers the loud
arm. The quiet arm is the hard case and ``xmod_agree`` is what stands in
front of it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.arrays import Array
from ..core.record import Access, Record, Scope, Severity, Verdict
from ..core.registry import Context, Direction, Layer, LayerSpec, register_layer

_WORD = re.compile(r"[a-z0-9]+")

#: Phrasings that direct a reader rather than describe a protein. Kept
#: deliberately short and readable: a long opaque pattern list invites
#: trust it has not earned.
INSTRUCTION_CUES = (
    "ignore previous", "ignore the above", "disregard", "instead of",
    "you must", "you should", "always return", "always output", "respond with",
    "answer with", "system prompt", "as an ai", "do not mention",
)
CONTACT_CUES = ("http://", "https://", "www.", "@", "contact ", "visit ")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@register_layer
class TextNearDuplicate(Layer):
    """A description that is a near-verbatim copy of one already present.

    Measured as the high-capture arm of the protein attack, and the one a
    filter catches by construction. The honest framing in the report: this
    layer raises the cost of the loud attack and does nothing to the quiet
    one.
    """

    spec = LayerSpec(
        name="text_nearduplicate",
        scope=Scope.CORPUS_RELATIVE,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"text"}),
        catches="Descriptions copied near-verbatim from an existing record or an expected "
                "query -- the arm measured at 0.58 capture templated, 0.97 free-text.",
        misses="A genuine paraphrase, which captures at 0.21 and shares few exact tokens. "
               "This layer is not evidence that a corpus is clean.",
        statistic="highest token Jaccard against the descriptions already in the corpus",
        direction=Direction.HIGH_IS_SUSPICIOUS,
        default_verdict=Verdict.QUARANTINE,
        severity=Severity.HIGH,
        default_params={"candidates": 200},
    )

    def score(self, records: Sequence[Record], ctx: Context) -> Array:
        adapter = ctx.adapter
        corpus_text = getattr(adapter, "texts", None) if adapter is not None else None
        if corpus_text is None:
            raise NotImplementedError("this corpus exposes no descriptions to compare with")
        texts = list(np.asarray(corpus_text, dtype=object))
        ids = [str(v) for v in np.asarray(adapter.ids, dtype=object)]
        out = np.zeros(len(records))
        for i, rec in enumerate(records):
            best = 0.0
            for other_id, other in zip(ids, texts):
                # a record already in the corpus must not match itself
                if other is None or other_id == rec.id:
                    continue
                best = max(best, jaccard(rec.text or "", str(other)))
            out[i] = best
        return out

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        return (
            f"This description shares {score:.0%} of its wording with one already in the "
            f"corpus. Near-verbatim copies are how a record wins retrieval for a request "
            f"it was written to intercept."
        )

    def remedy(self, record: Record) -> str:
        return "Check whether this duplicates an existing entry; if it is a new protein, "\
               "describe it in its own words."


@register_layer
class TextInstruction(Layer):
    """Instruction-shaped content inside a scientific description.

    A retrieved description is read by a model, so text that addresses the
    reader rather than describing the protein is out of place regardless
    of whether it is adversarial.
    """

    spec = LayerSpec(
        name="text_instruction",
        scope=Scope.RECORD_LOCAL,
        access=frozenset({Access.INSERT, Access.OVERWRITE}),
        needs=frozenset({"text"}),
        catches="Imperatives aimed at the reader, URLs and contact strings inside what "
                "should be a description of a protein.",
        misses="Poison that reads entirely like biology, which is the competent version "
               "of this attack.",
        statistic="count of instruction-shaped and contact-shaped phrases",
        direction=Direction.HIGH_IS_SUSPICIOUS,
        default_verdict=Verdict.QUARANTINE,
        severity=Severity.HIGH,
        budgeted=False,
        default_params={"threshold": 0.5},
    )

    def __init__(self, threshold: float | None = None, **kw: Any) -> None:
        super().__init__(threshold if threshold is not None else 0.5, **kw)

    def score(self, records: Sequence[Record], ctx: Context) -> Array:
        return np.array([float(len(self._hits(r.text or ""))) for r in records])

    @staticmethod
    def _hits(text: str) -> list[str]:
        low = text.lower()
        return [c for c in INSTRUCTION_CUES + CONTACT_CUES if c in low]

    def evidence(self, record: Record, index: int, ctx: Context) -> dict[str, Any]:
        return {"matched": self._hits(record.text or "")}

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        matched = ", ".join(repr(m) for m in evidence.get("matched", [])[:4])
        return (
            f"This description contains wording that addresses a reader rather than "
            f"describing a protein ({matched}). A corpus entry should not give instructions."
        )

    def remedy(self, record: Record) -> str:
        return "Reject, and check the rest of the submission from the same source."


@register_layer
class TextFluency(Layer):
    """Descriptions whose wording does not look like the rest of the corpus.

    Gradient-optimised hub text is typically fluent to a retriever and odd
    to a reader. The statistic is a unigram surprisal against the corpus's
    own vocabulary -- a surrogate for a language model, chosen so the
    layer carries no model dependency. It is weaker than an LM
    perplexity and the report says so.
    """

    spec = LayerSpec(
        name="text_fluency",
        scope=Scope.CORPUS_RELATIVE,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"text"}),
        catches="Token-salad and optimised hub descriptions built from words the corpus "
                "rarely uses.",
        misses="Fluent paraphrases, and optimised text that reuses common vocabulary. A "
               "unigram surrogate, not a language model.",
        statistic="mean unigram surprisal of the description under the corpus vocabulary",
        direction=Direction.HIGH_IS_SUSPICIOUS,
        default_verdict=Verdict.FLAG,
        severity=Severity.MEDIUM,
    )

    def _model(self, ctx: Context) -> tuple[Counter[str], int]:
        if "unigram" not in ctx.cache:
            counter: Counter[str] = Counter()
            corpus_text = getattr(ctx.adapter, "texts", None)
            if corpus_text is None:
                raise NotImplementedError("this corpus exposes no descriptions")
            for t in np.asarray(corpus_text, dtype=object):
                if t is not None:
                    counter.update(tokens(str(t)))
            ctx.cache["unigram"] = (counter, max(sum(counter.values()), 1))
        return ctx.cache["unigram"]  # type: ignore[no-any-return]

    def score(self, records: Sequence[Record], ctx: Context) -> Array:
        counter, total = self._model(ctx)
        vocab = len(counter) + 1
        out = np.zeros(len(records))
        for i, rec in enumerate(records):
            toks = tokens(rec.text or "")
            if not toks:
                out[i] = 0.0
                continue
            out[i] = float(
                np.mean([
                    -math.log((counter.get(t, 0) + 1) / (total + vocab)) for t in toks
                ]).item()
            )
        return out

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        return (
            f"The wording of this description is unusual for this corpus (mean surprisal "
            f"{score:.2f}). Check it reads like the others before accepting."
        )


@register_layer
class FragmentOutlier(Layer):
    """Payload fragments far outside the corpus's own distribution.

    In ProDVa the retrieved record's fragments enter the generator's
    dynamic vocabulary, and upstream applies no length cap at inference.
    The delivery channel is therefore the fragment list, which makes its
    shape worth checking even though a competent payload will look normal.
    """

    spec = LayerSpec(
        name="fragment_outlier",
        scope=Scope.RECORD_LOCAL,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"payload"}),
        catches="Abnormally long or numerous fragments entering the dynamic vocabulary.",
        misses="A payload with ordinary fragment statistics, which is easy to arrange.",
        statistic="fragment count and longest fragment, as a robust z against the corpus",
        direction=Direction.HIGH_IS_SUSPICIOUS,
        default_verdict=Verdict.FLAG,
        severity=Severity.MEDIUM,
    )

    @staticmethod
    def _stats(rec: Record) -> tuple[float, float]:
        frags = rec.features.get("fragments") if rec.features else None
        if not frags:
            return 0.0, 0.0
        lengths = [len(str(f)) for f in frags]
        return float(len(frags)), float(max(lengths))

    def score(self, records: Sequence[Record], ctx: Context) -> Array:
        table = np.array([self._stats(r) for r in records], dtype=float)
        ref = ctx.cache.get("fragment_reference", table)
        med = np.median(ref, axis=0)
        mad = np.median(np.abs(ref - med), axis=0)
        mad = np.where(mad <= 0, 1.0, mad)
        z = np.abs(table - med) / (1.4826 * mad)
        return np.asarray(z.max(axis=1), dtype=np.float64)

    def reason(self, record: Record, score: float, evidence: dict[str, Any]) -> str:
        n, longest = self._stats(record)
        return (
            f"This record contributes {int(n)} fragments, the longest {int(longest)} "
            f"residues, which is {score:.1f} robust standard deviations from the corpus."
        )
