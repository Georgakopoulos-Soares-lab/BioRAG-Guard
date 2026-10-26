# 2. The cross-modal agreement backend is pluggable and has no biological default

**Status:** accepted, pending a deployment decision

## Context

In the protein arm, a payload attached to an *unrelated* description
captured 0 of 100 queries, while the same payload behind a targeted
description captured up to 0.97. The payload is not what wins retrieval;
the bond between a plausible description and an unrelated payload is the
attack. Verifying that bond is therefore the highest-value check on a
text-keyed corpus, and the only one with purchase on the genuinely
paraphrased arm, which evades near-duplicate detection by construction and
still captures at 0.21.

Scoring it properly needs an annotation source: profile-HMM search against
the payload's domains, InterProScan, or a text-protein embedding model.
Each is a different dependency and a different deployment cost, and the
choice is the corpus owner's.

## Decision

`crossmodal_agreement` takes a backend. The shipped default,
`term_overlap`, compares the description against annotation terms the
**adapter already read out of the corpus** -- for ProDVa, the `name`,
`type` and `description` fields of the fragment map. It derives no biology
from the sequence; it checks two assertions the corpus already makes
against each other.

When no annotation terms are available the layer raises, and the report
says the backend had nothing to score.

## Consequences

* The ProDVa extra stays light. No HMM search, no InterProScan, no
  transformer in the default install.
* The honest limitation is explicit: with the shipped backend the layer
  checks *internal consistency*, which catches a description welded to a
  differently-annotated payload and does not catch a payload whose
  annotations were fabricated to match. A real annotation backend closes
  that gap and is a deployment decision.
* Silence from this layer is never a pass. That is enforced by a test.
