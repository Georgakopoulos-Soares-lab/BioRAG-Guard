# 3. One `hub` layer, not one per modality

**Status:** accepted

## Context

The specification listed `POP-HUBCELL` and `POP-HUBDESC` as separate
checks, one for cell corpora and one for text corpora.

## Decision

One layer, `hub`. The statistic is identical -- k-occurrence over a probe
set -- and the modality only changes where the probes come from, which is
the adapter's job via `probe_queries`.

## Consequences

* One name in the config, the report, the docs and the code, which is the
  ubiquitous-language rule.
* An adapter that wants a different probe distribution overrides
  `probe_queries` rather than needing its own layer.
* The report names the corpus, so a reader always knows whether a hub is a
  cell or a description.
