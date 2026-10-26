# BioRAG-Guard

Poisoning defences for biological retrieval corpora. You point it at a
corpus and it answers one question: **has anything been planted in here,
and should I accept this new record?**

```bash
pip install biorag-guard
bioguard demo            # the whole flow on a synthetic corpus, in a second
```

## Two jobs

They are not two views of the same mechanism. They answer the two threat
models the underlying research separates, and the defences barely overlap.

| | **Guard at ingest** &mdash; `bioguard check` | **Guard on demand** &mdash; `bioguard scan` |
|---|---|---|
| threat | the attacker can only get records **accepted** | the attacker can **edit** records already present |
| works because | a submission can be screened against the corpus as it stood before it | an edit changes state that can be compared with a manifest |
| blind to | hub records, and every in-place edit | nothing an ingest gate sees, but it runs on a schedule rather than per record |

## Three commands

```bash
bioguard init /data/scimilarity_model_v1.1       # detect the corpus, write a config
bioguard manifest create /data/scimilarity_model_v1.1
bioguard calibrate /data/scimilarity_model_v1.1 --budget 0.05
bioguard scan /data/scimilarity_model_v1.1
```

No config file has to be written by hand. Every threshold comes from a
**budget** &mdash; the fraction of genuine records a layer is allowed to
sacrifice &mdash; measured on a held-out split, and `calibrate` prints
what that cost on clean data.

## Five lines in Python

```python
from biorag_guard import Guard, Verdict

guard = Guard.load("bioguard.yaml")        # or Guard.auto("/data/model_v1.1")

result = guard.check(record)
if result.verdict is Verdict.ADMIT:
    index.add(record)
```

Also available: `guard.check_many(records)`, the `@guard.gate` decorator,
`with guard.gating(index) as safe_index`, `guard.scan()`, and
`guard.vote(query)` for the query-time policy.

## What it checks, and what it does not

The layers are ported from measured research, not invented, and each one
carries the attack it stops **and** the attack it does not:

| layer | stops | does not stop |
|---|---|---|
| `recompute` | synthesised geometry: interpolated vectors, cluster floods, re-synthesised edits | a relabelled or duplicated real cell &mdash; its vector re-derives perfectly |
| `label_consistency` | a genuine embedding submitted under a label its neighbours do not share | a label rewritten on a record already in the corpus |
| `dedup` | verbatim and near-verbatim copies, reading no metadata at all | a flood of distinct synthesised records; unsound at query time |
| `donor_cap` | influence concentrated in copies from one source | fabricated donor identities; measured as *harmful* against in-place relabelling |
| `manifest_label` | labels rewritten in place &mdash; the attack no vote-time layer touches | edits made before the first manifest |
| `hub` | records positioned to be returned for many unrelated queries | targeted poison aimed at a single query |

`recompute` and `dedup`/`label_consistency` are complementary by
construction: each is blind exactly where the other works. **A policy that
enables one family has a hole in it**, which is why the defaults enable
both.

Run `bioguard explain <layer>` for the full statement, including the
calibrated threshold on your corpus.

## The result that shaped the design

Against an attacker who rewrites the label of a record already in the
corpus, every vote-time defence measured no effect &mdash; and the donor
cap made things worse. What catches it is arithmetic over the label
column, and over a reference of 7,913,892 records it is almost free:

| check | detects an in-place relabel | wall time |
|---|---|---|
| hash of the index | **no** | 1.37 s |
| hash of the label column | yes | 0.10 s |
| element-wise label diff | yes | 0.31 s |

A checksum over the vectors cannot see a relabel, because the vectors do
not move. That is why `bioguard manifest create` is the first thing the
tool asks you to run.

## Corpora

| adapter | corpus | extra |
|---|---|---|
| `scimilarity` | a SCimilarity cell-embedding reference atlas | `biorag-guard[scimilarity]` for the shipped hnswlib index |
| `prodva` | a description-keyed protein corpus (data file, sequence map, fragment map) | &mdash; |
| `generic` | vectors in a `.npy`, records in a `.jsonl` | &mdash; |

Both first-class adapters work **without model weights**: everything
except `recompute` needs only the precomputed embeddings and the
metadata. A new corpus is a `CorpusAdapter` subclass of about a hundred
lines &mdash; see [docs/extending.md](docs/extending.md), or run
`bioguard new-adapter`.

## Honest limits

* Every layer here is **evadable by an adaptive attacker**, and two of
  them have measured blind spots listed above. The tool raises the cost of
  the loud attacks and surfaces the population-level ones; it does not
  make a corpus safe.
* It detects **manipulation of a retrieval corpus**. It does not judge
  whether a sequence or a cell is hazardous and must not be described as
  doing so. Use dedicated biosecurity screening for that question.
* It is not an attack toolkit. The synthetic fixtures in
  `biorag_guard.demo` exist to test defences; no poisoned index, corpus or
  mapping file is produced or committed.

## Provenance

The cell-corpus layers are ported from
[Single_Cell_Poisoning](https://github.com/Georgakopoulos-Soares-lab/Single_Cell_Poisoning)
(MIT), keeping their names and semantics; `tests/` runs the original
implementations side by side with the ports and asserts they agree. The
text-protein layers are informed by
[prodva_rag_poisoning](https://github.com/Georgakopoulos-Soares-lab/prodva_rag_poisoning).

MIT licensed. Security issues: see [SECURITY.md](SECURITY.md).
