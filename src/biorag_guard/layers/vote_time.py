"""Layers that exist only at query time.

``donor_cap``, ``weight``, ``abstain`` and the seasoning window are not
filters on an arriving record and not sweeps over a corpus: they change how
a retrieved neighbourhood is turned into an answer. They are registered
here so ``bioguard explain`` can describe them and so the report can say
plainly that they did not run during ``check`` or ``scan``.

They are configured under ``vote:`` and applied by ``Guard.vote``.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.finding import Finding
from ..core.record import Access, Record, Scope, Severity, Verdict
from ..core.registry import Context, Layer, LayerSpec, register_layer


class _VoteTimeLayer(Layer):
    """Describable, explainable, and explicitly not runnable in a sweep."""

    def run(self, records: Sequence[Record], ctx: Context) -> list[Finding]:
        raise NotImplementedError(
            f"{self.name} applies at query time; configure it under `vote:` "
            "and call guard.vote()"
        )


@register_layer
class DonorCap(_VoteTimeLayer):
    """At most ``cap`` votes per donor in a retrieved neighbourhood.

    Measured: takes ``duplicate`` and ``relabel`` from 0.343 to 0.010 at
    cap 1. Against an attacker who overwrites records it is *worse than
    nothing* -- 0.178 against 0.162 undefended -- and its whole mechanism
    collapses if the attacker fabricates donor identities, which is what
    the research's identity sweep measures.
    """

    spec = LayerSpec(
        name="donor_cap",
        scope=Scope.VOTE_TIME,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"donor"}),
        catches="Influence concentrated in copies: many records from one source crowding "
                "a single neighbourhood.",
        misses="Fabricated donor identities, which cost an attacker nothing. Measured as "
               "actively harmful against in-place relabelling.",
        statistic="votes retained per donor within one query's neighbourhood",
        ported=True,
        budgeted=False,
        default_verdict=Verdict.FLAG,
        severity=Severity.MEDIUM,
        default_params={"cap": 1},
    )


@register_layer
class VoteWeight(_VoteTimeLayer):
    """Weight neighbours by proximity instead of one-vote-each.

    The one policy here that is not a filter: nothing is rejected, the
    arithmetic changes. It may *amplify* a cluster flood, which places its
    records closer to the query than genuine neighbours are. The research
    measures it only in composition, never standalone, because standalone
    is the configuration where poison a record layer would have rejected is
    still voting.
    """

    spec = LayerSpec(
        name="weight",
        scope=Scope.VOTE_TIME,
        access=frozenset({Access.INSERT, Access.OVERWRITE}),
        needs=frozenset(),
        catches="Nothing on its own. Measured only as a modifier on a filtered "
                "neighbourhood.",
        misses="Everything, considered as a detector. Can make a cluster flood stronger.",
        statistic="uniform | score (1/(1-cos)) | cosine | rank | softmax",
        ported=True,
        budgeted=False,
        default_verdict=Verdict.FLAG,
        severity=Severity.INFO,
        default_params={"scheme": "uniform", "temp": 0.01},
    )


@register_layer
class Abstain(_VoteTimeLayer):
    """Decline on a thin margin rather than guess.

    Declining is a legitimate outcome for an annotation system and is not
    counted as an error. Report the abstention rate beside the accuracy or
    the comparison is meaningless.
    """

    spec = LayerSpec(
        name="abstain",
        scope=Scope.VOTE_TIME,
        access=frozenset({Access.INSERT, Access.OVERWRITE}),
        needs=frozenset(),
        catches="Converts thin-margin wrong answers into refusals.",
        misses="Not a detector. A confident poisoned majority is answered confidently.",
        statistic="top-label vote share minus runner-up share",
        ported=True,
        budgeted=False,
        default_verdict=Verdict.FLAG,
        severity=Severity.INFO,
        default_params={"min_margin": 0.0},
    )


@register_layer
class Seasoning(_VoteTimeLayer):
    """A record newer than the window does not vote.

    Only meaningful against insertion: an edited record has been in the
    corpus all along and is seasoned by definition.
    """

    spec = LayerSpec(
        name="seasoning",
        scope=Scope.VOTE_TIME,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"added_at"}),
        catches="Freshly inserted records influencing answers before anyone has reviewed "
                "them.",
        misses="Anything already past the window, and every in-place edit.",
        statistic="age of the record against the quarantine window",
        ported=True,
        budgeted=False,
        default_verdict=Verdict.FLAG,
        severity=Severity.LOW,
        default_params={"days": 30},
    )
