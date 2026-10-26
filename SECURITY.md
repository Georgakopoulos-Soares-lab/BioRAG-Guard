# Security policy

## Reporting a bypass

If you find a way past a layer, please report it privately to the
maintainers rather than opening an issue. Expect an acknowledgement within
a week.

Evasion techniques are not published in the issue tracker. Describing what
a layer measures is necessary and is already in the documentation;
describing how to tune poison past it is not.

## What this tool does and does not do

BioRAG-Guard detects **manipulation of a retrieval corpus**: records
inserted to be retrieved, records edited after the fact, and corpora whose
files have drifted out of agreement.

It does **not** assess whether a sequence, a cell, or any other payload is
hazardous. It is not a biosecurity screen and must not be presented as
one. For that question, use dedicated screening and your institution's
biosafety process.

## Every layer is evadable

This is a statement about the design, not a disclaimer. Each layer targets
a measured attack shape and each has a blind spot that is documented in
`bioguard explain <layer>` and in the README table. An attacker who knows
which layers are enabled can construct poison that satisfies all of them.
What the tool buys is cost, coverage of the loud arms, and a population
view that per-record screening structurally cannot provide.

The honest summary of the evidence: against an attacker who can only
submit records, the layers are effective and cheap. Against one who can
edit the corpus in place, the only thing that works is integrity
monitoring.

## Data handling

No poisoned index, corpus, label file or mapping file is committed to this
repository. The fixtures in `biorag_guard.demo` are synthetic and exist to
exercise the defences.
