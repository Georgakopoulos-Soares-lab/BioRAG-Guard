# 1. Two jobs, split along the threat model

**Status:** accepted

## Context

The obvious design is one detector applied in two places. The
measurements say that is wrong.

Against an attacker who can only submit records, the vote-time and
submission-time layers are effective: `relabel` and `duplicate` fall from
0.343 to 0.001-0.002, `flood` from 0.264 to 0.001-0.004.

Against an attacker who overwrites records already in the corpus, the same
layers do nothing. For `unmoved_inplace` -- the label rewritten, the
embedding untouched -- undefended attack success is 0.162, label
consistency gives 0.149, the full stack 0.150, and the donor cap is
*worse* at 0.178.

## Decision

Two jobs, named after what they protect against rather than when they run:

* `bioguard check` serves the **insert** model. It runs record-local and
  corpus-relative layers, and refuses a batch that is already indexed,
  because `label_consistency` must score against the corpus as it stood
  before the batch.
* `bioguard scan` serves the **overwrite** model and everything that is
  only visible from the population. Its most important layer is the
  manifest comparison, not a detector.

## Consequences

* A user running only `check` is told, in the report, that hub records and
  in-place edits are invisible to it.
* `manifest create` is milestone-zero work, ahead of any detector: over
  7.9M records a label hash costs 0.10 s and catches what nothing else
  does, while an index hash costs 1.37 s and catches nothing here.
* The vote is a third surface rather than part of either job, because
  `donor_cap`, `weight` and `abstain` have no ingest-time form.
