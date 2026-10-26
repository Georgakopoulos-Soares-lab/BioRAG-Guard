"""Generate the layer reference from the registry.

A new layer is documented by existing. Keeping the reference page as
prose would let it drift from the code, and what a layer misses is
exactly the sentence that would go stale first.
"""

from __future__ import annotations

from pathlib import Path

from .core.record import Scope
from .core.registry import all_layers
from .layers import cell  # noqa: F401  (import registers every layer)

_SCOPE_NOTE = {
    Scope.RECORD_LOCAL: "needs only the record; runs at ingest and in a sweep",
    Scope.CORPUS_RELATIVE: "needs one neighbour query against the corpus; runs at ingest "
                           "and in a sweep",
    Scope.CORPUS_WIDE: "needs the whole corpus; **a sweep only** -- ingest screening cannot "
                       "see this",
    Scope.VOTE_TIME: "applies when a query is answered; configure under `vote:`",
}


def render_layer_reference() -> str:
    lines = [
        "# Layer reference",
        "",
        "Generated from the registry by `python -m biorag_guard.docs_gen`. Every field",
        "below is data on the layer, which is also what `bioguard explain <layer>` prints.",
        "",
        "**Ported** means the implementation comes from the research code and keeps its",
        "semantics; `tests/` runs the originals beside the ports and asserts they agree.",
        "",
        "| layer | scope | ported | catches | misses |",
        "|---|---|---|---|---|",
    ]
    for name, cls in all_layers().items():
        s = cls.spec
        lines.append(
            f"| [`{name}`](#{name.replace('_', '-')}) | {s.scope.value} | "
            f"{'yes' if s.ported else 'no'} | {s.catches} | {s.misses} |"
        )
    lines.append("")

    for name, cls in all_layers().items():
        s = cls.spec
        lines += [
            f"## {name}",
            "",
            f"*{_SCOPE_NOTE[s.scope]}*",
            "",
            f"**Measures.** {s.statistic}.",
            "",
            f"**Catches.** {s.catches}",
            "",
            f"**Misses.** {s.misses}",
            "",
            f"- threat model: {', '.join(sorted(a.value for a in s.access))}",
            f"- needs: {', '.join(sorted(s.needs)) or 'nothing beyond the record'}",
            f"- threshold: {'from the budget, via `bioguard calibrate`' if s.budgeted else 'fixed'}",  # noqa: E501
            f"- default verdict: `{s.default_verdict.value}`",
        ]
        if s.ingest_only:
            lines.append("- **sound at ingest only** -- at query time an exact copy ties "
                         "with its original and the tie-break decides which survives")
        if s.default_params:
            params = ", ".join(f"`{k}={v!r}`" for k, v in s.default_params.items())
            lines.append(f"- parameters: {params}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    target = Path("docs/layers.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_layer_reference())
    print(f"wrote {target} for {len(all_layers())} layers")


if __name__ == "__main__":
    main()
