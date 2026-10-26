# Contributing

## Adding a layer

A layer is one defence, switchable alone. It declares what it needs, what
it catches, and **what it misses** &mdash; the last is not optional and is
enforced by a test.

```python
@register_layer
class MyLayer(Layer):
    spec = LayerSpec(
        name="my_layer",
        scope=Scope.CORPUS_RELATIVE,
        access=frozenset({Access.INSERT}),
        needs=frozenset({"vector", "label"}),
        catches="the attack shape this stops, in one sentence",
        misses="what it provably does not stop. Never leave this empty.",
        statistic="the quantity computed, in one line",
        direction=Direction.LOW_IS_SUSPICIOUS,
        default_verdict=Verdict.QUARANTINE,
    )

    def score(self, records, ctx):
        return np.array([...])          # one score per record

    def reason(self, record, score, evidence):
        return "a sentence a bench scientist understands"
```

A scorer gets its threshold, its calibration and its reporting for free.
Override `run` instead only when the layer is not a scorer, as the
integrity and set-algebra layers are.

Rules that are enforced:

* **a missing input raises, it never passes.** Reporting a layer that
  never ran as a layer that found nothing is the one failure mode that
  cannot be recovered downstream;
* `reason` is written for a bench scientist. Not "LOF 2.31 exceeds
  threshold" but "this cell's 30 nearest neighbours are 2.3x further away
  than is typical for this tissue";
* no number appears in a docstring, a report or the documentation unless
  it came from a run.

## Adding an adapter

Subclass `CorpusAdapter`, implement `sniff`, `open`, `ids`, and whichever
of `vectors` / `labels` / `donors` / `texts` the corpus has, then register
it under the `biorag_guard.adapters` entry-point group. Target a hundred
lines. `docs/extending.md` walks through one.

## Ported code

Layers marked `ported=True` come from the research implementations and
must keep their semantics, including the awkward parts: self-matches
excluded from label consistency, an empty neighbourhood scoring 1.0,
`dedup` sound at ingest only, score weights clipped at `SCORE_EPS`. The
tests in `tests/test_vote_equivalence.py` and
`tests/test_label_consistency_equivalence.py` run the originals beside the
ports and assert equality; if you change a ported layer, those tests are
the specification.

## Before opening a pull request

```bash
ruff check src tests
mypy
pytest
```
