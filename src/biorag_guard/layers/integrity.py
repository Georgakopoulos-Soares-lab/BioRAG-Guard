"""Manifest layers: catch the edit that no vote-time filter can.

Against an attacker who rewrites the label of a record already in the
corpus, the research measured every defence layer at no effect -- 0.149 to
0.178 against 0.162 undefended, with the donor cap the worst of them. The
thing that does work is arithmetic over the label column, and it is almost
free: over a 7,913,892-record reference, hashing the labels took 0.10 s and
an element-wise diff 0.31 s, while hashing the index took 1.37 s and
**did not detect the relabel at all**, because the vectors never moved.

Two layers, because they catch different things and a user needs to know
which one fired.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..core.finding import Finding
from ..core.manifest import Manifest, ManifestDiff, compare
from ..core.record import Access, Record, Scope, Severity, Verdict
from ..core.registry import Context, Layer, LayerSpec, register_layer


def manifest_diff(ctx: Context, path: str | Path) -> ManifestDiff:
    """Compare the stored manifest with the corpus now. Cached per run."""
    key = f"manifest_diff:{path}"
    if key not in ctx.cache:
        before = Manifest.load(path)
        after = ctx.adapter.manifest(keep_labels=True)
        ctx.cache[key] = compare(before, after)
    return ctx.cache[key]  # type: ignore[no-any-return]


class _ManifestLayer(Layer):
    def _path(self) -> Path:
        p = self.params.get("path")
        if not p:
            raise NotImplementedError("no manifest path configured")
        path = Path(p)
        if not path.exists():
            raise NotImplementedError(
                f"no manifest at {path}; run `bioguard manifest create` first"
            )
        return path


@register_layer
class ManifestIndex(_ManifestLayer):
    """Have the stored vectors changed since the manifest was taken?

    Fast and complete for anything that moves a vector. Blind to a
    relabel, and the report says so next to the result so a clean index
    hash is never read as a clean corpus.
    """

    spec = LayerSpec(
        name="manifest_index",
        scope=Scope.CORPUS_WIDE,
        access=frozenset({Access.OVERWRITE}),
        needs=frozenset(),
        catches="Any edit that changes a stored vector: shifted or re-synthesised "
                "embeddings, and records inserted into the middle of the index.",
        misses="A relabel. The label can be rewritten with the index byte-identical, and "
               "this check will pass. That is measured, not hypothetical.",
        statistic="blake2b over the vector matrix, plus one digest per 100,000-record block",
        ported=True,
        budgeted=False,
        default_verdict=Verdict.REJECT,
        severity=Severity.HIGH,
        default_params={"path": "bioguard_manifest.json"},
    )

    def run(self, records: Sequence[Record], ctx: Context) -> list[Finding]:
        diff = manifest_diff(ctx, self._path())
        if not diff.index_changed:
            return []
        blocks = diff.changed_index_blocks
        where = (f" in {len(blocks)} block(s) of 100,000 records, first at block {blocks[0]}"
                 if blocks else "")
        return [
            Finding(
                layer=self.name, record_id="<corpus>", score=float(len(blocks)),
                threshold=0.0, severity=self.spec.severity, verdict=self.verdict,
                reason=(f"The stored vectors have changed since the manifest of "
                        f"{diff.n_before:,} records{where}."),
                remedy="Compare against your own backup of the index before trusting any "
                       "answer from this corpus.",
                evidence=diff.to_dict(),
            )
        ]


@register_layer
class ManifestLabel(_ManifestLayer):
    """Have any labels changed on records that kept their embedding?

    This is the layer that catches the attack nothing else does. It also
    names the records, because knowing that something changed is much less
    useful than knowing which records moved and where they moved to.
    """

    spec = LayerSpec(
        name="manifest_label",
        scope=Scope.CORPUS_WIDE,
        access=frozenset({Access.OVERWRITE}),
        needs=frozenset({"label"}),
        catches="Labels rewritten in place -- the one attack on which every vote-time "
                "layer measured no effect. Names the records and where they moved to.",
        misses="Edits made before the first manifest was taken. The manifest is a "
               "starting point, not a history.",
        statistic="blake2b over the label column, then an id-joined element-wise diff",
        ported=True,
        budgeted=False,
        default_verdict=Verdict.REJECT,
        severity=Severity.HIGH,
        default_params={"path": "bioguard_manifest.json"},
    )

    def run(self, records: Sequence[Record], ctx: Context) -> list[Finding]:
        diff = manifest_diff(ctx, self._path())
        if not diff.labels_changed:
            return []
        tally = diff.destination_tally()
        top = next(iter(tally.items()), None)
        concentration = (
            f" {top[1]:,} of them now carry the label {top[0]!r}." if top else ""
        )
        silent = (
            " The index hash is unchanged, so a checksum over the vectors would have "
            "reported this corpus as clean." if diff.relabel_in_place else ""
        )
        n = diff.n_labels_changed
        counted = f"{n:,} records have" if n else "Labels have"
        return [
            Finding(
                layer=self.name, record_id="<corpus>", score=float(n), threshold=0.0,
                severity=self.spec.severity, verdict=self.verdict,
                reason=(f"{counted} a different label than when the manifest was "
                        f"taken.{concentration}{silent}"),
                remedy="Restore the labels from a trusted copy, or confirm the change with "
                       "whoever curates the corpus before answering any more queries.",
                evidence=diff.to_dict(),
            )
        ]
