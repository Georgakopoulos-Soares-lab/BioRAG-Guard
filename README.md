<div align="center">

# 🧬 BioRAG-Guard

**Integrity checks for biological retrieval corpora.**
Screen new records. Audit existing collections.

[![ci](https://github.com/biosafelabs/BioRAG-Guard/actions/workflows/ci.yml/badge.svg)](https://github.com/biosafelabs/BioRAG-Guard/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A research tool from [BioSafeLabs](https://biosafelabs.org).

</div>

---

## 🔬 Why a retrieval corpus needs guarding

A growing class of biological tools answers questions by **looking things
up**, not by reasoning about them. The answer is whatever came back from a
database of vectors.

**Single cell.** A model such as [SCimilarity](https://github.com/Genentech/scimilarity)
turns one cell's expression profile into a 128-number vector. To find out
what kind of cell it is, you search a reference atlas — 7.9 million
annotated cells — for the nearest vectors and read off the labels they
carry. The atlas *is* the answer. Change the labels on a few hundred of
its rows and you change what a lab concludes about a cell type, with the
model itself untouched.

**Protein.** A text-keyed protein corpus such as
[ProDVa](https://github.com/sornkL/ProDVa) takes a plain-English
description of a function and retrieves real sequence fragments matching
it. Those fragments go straight into a designed protein. Whoever controls
which fragments are returned for *"thermostable hydrolase"* has a hand on
the design.

**Why it is worth attacking.** Changing a model means retraining it.
Changing a retrieval corpus means editing a row. These references are
assembled from public submissions, shipped as downloadable tarballs, and
copied into labs where nobody re-derives them from scratch. In the
measurements this tool is built on, 1,500 planted records were enough to
be returned for 60% of free-text protein queries — and on the single-cell
side the cheapest effective edit changes no number in the file at all,
only a label.

BioRAG-Guard answers one question about such a corpus: **has anything been
planted in here, and should I accept this new record?**

> ⚠️ It detects *manipulation of a corpus*. It does **not** judge whether a
> sequence or a cell is hazardous, and must not be described as doing so.
> See [SECURITY.md](SECURITY.md), and "Honest limits" below.

---

## ⚡ Install

```bash
pip install biorag-guard
bioguard demo            # the whole flow on a synthetic corpus, in a second
```

`demo` needs no downloads, no data and no model weights. It builds a small
synthetic corpus, takes a manifest, calibrates, screens a batch per attack
shape, then relabels records in place and shows which checks notice.

---

## 🛡️ Two modes

Not two views of the same mechanism. They answer the two threat models the
underlying research keeps apart, and the defences barely overlap.

| | 📥 **Check at ingest** — `bioguard check` | 🔍 **Scan a stored corpus** — `bioguard scan` |
|---|---|---|
| **the attacker can** | only get records **accepted** into the corpus | **edit** records already present |
| **works because** | a submission can be screened against the corpus as it stood *before* it arrived | an edit changes state that can be compared against a saved manifest |
| **runs** | per batch, in your ingest pipeline | on a schedule, or on demand |
| **blind to** | hub records, and every in-place edit made later | anything changed before the first manifest was taken |

A third surface, `guard.vote`, applies the query-time policy — see
[docs/two-modes.md](docs/two-modes.md).

---

## ⌨️ Four commands

```bash
CORPUS=/data/scimilarity_model_v1.1

bioguard init             $CORPUS            # detect the layout, write a config
bioguard manifest create  $CORPUS            # snapshot it while you still trust it
bioguard calibrate        $CORPUS --budget 0.05
bioguard scan             $CORPUS            # audit the whole corpus
```

and to screen an arriving batch **before** it is indexed:

```bash
bioguard check new_study/ --corpus $CORPUS
```

Exit codes: `0` nothing above a flag, `1` a quarantine, `2` a reject, `3` a
run error — so it drops into CI unchanged.

No config file has to be written by hand. Every threshold comes from a
**budget** — the fraction of genuine records a layer is allowed to
sacrifice — measured on a held-out split. `calibrate` prints what that
cost on clean data, so you see the price before you pay it.

### 🐍 The same thing in Python

```python
from biorag_guard import Guard, Verdict

guard = Guard.load("bioguard.yaml")        # or Guard.auto("/data/model_v1.1")

result = guard.check(record)
if result.verdict is Verdict.ADMIT:
    index.add(record)
```

Also available: `guard.check_many(records)`, the `@guard.gate` decorator,
`with guard.gating(index) as safe_index`, `guard.scan()`, and
`guard.vote(query)` for the query-time policy. Over HTTP, for non-Python
pipelines: `bioguard serve`.

---

## 🧱 What it checks, and what it does not

Every layer is ported from or informed by measured research, and each one
carries the attack it stops **and** the attack it does not:

| | layer | stops | does not stop |
|---|---|---|---|
| 🧮 | `recompute` | synthesised geometry: interpolated vectors, cluster floods, re-synthesised edits | a relabelled or duplicated *real* cell — its vector re-derives perfectly |
| 🏷️ | `label_consistency` | a genuine embedding submitted under a label its neighbours do not share | a label rewritten on a record already in the corpus |
| 📑 | `dedup` | verbatim and near-verbatim copies, reading no metadata at all | a flood of distinct synthesised records; unsound at query time |
| 🎚️ | `donor_cap` | influence concentrated in copies from one source | fabricated donor identities; measured as *harmful* against in-place relabelling |
| 📋 | `manifest_label` | labels rewritten in place — the attack no vote-time layer touches | edits made before the first manifest |
| 🕸️ | `hub` | records positioned to be returned for many unrelated queries | targeted poison aimed at a single query |

`recompute` and `dedup`/`label_consistency` are complementary by
construction: each is blind exactly where the other works. **A policy that
enables only one family has a hole in it**, which is why the defaults
enable both.

Run `bioguard explain <layer>` for the full statement, including the
calibrated threshold on your corpus, or read
[docs/layers.md](docs/layers.md) for all of them.

---

## 🧾 The measurement that decided the first command

This is the result that shaped the tool, so it is worth stating without
shorthand.

**The attack.** Someone with write access to the corpus picks a record and
changes one thing: its label. The vector — the measured expression profile
— is left exactly as it was. The record is still a real cell, still sitting
in the same place in the index. Only the answer it votes for has changed.

**What the measurements say.** Every filter in the tool was run against
that edit on the single-cell reference, and compared against doing nothing
at all. Lower attack success is better:

| configuration | attack success |
|---|---|
| no defence | 0.162 |
| `label_consistency` | 0.149 |
| `dedup` | 0.163 |
| `recompute` | 0.162 |
| all of them together | 0.150 |
| `donor_cap` | **0.178** ← worse than no defence |

Nothing helped, and the per-source vote cap actively hurt. That is not a
tuning failure. All of these filters ask some version of *"does this
record's vector look out of place?"* — and the vector is genuine and has
not moved a millimetre.

**What does work** is keeping a copy of the label column from when the
corpus was trusted, and comparing. On the 7,913,892-record reference:

| check | catches an in-place relabel? | wall time |
|---|---|---|
| checksum of the whole index | ❌ no | 1.37 s |
| checksum of the label column | ✅ yes | 0.10 s |
| element-wise diff of the label column | ✅ yes | 0.31 s |

A checksum over the vectors cannot see a relabel, because the vectors do
not move — and it costs **fourteen times more** than the check that works.

The catch is that a comparison needs something to compare against. If the
first snapshot is taken after the edit, the edit is already baked in. That
is why `bioguard manifest create` is the first thing the tool asks you to
run, ahead of any detector.

---

## 🗂️ Corpora

| adapter | corpus | source | extra |
|---|---|---|---|
| 🧫 `scimilarity` | a SCimilarity cell-embedding reference atlas | [Genentech/scimilarity](https://github.com/Genentech/scimilarity) · [weights on Zenodo](https://zenodo.org/records/10685499) | `biorag-guard[scimilarity]` for the shipped hnswlib index |
| 🧪 `prodva` | a description-keyed protein corpus (data file, sequence map, fragment map) | [sornkL/ProDVa](https://github.com/sornkL/ProDVa) | — |
| 📦 `generic` | vectors in a `.npy`, records in a `.jsonl` | — | — |

Both first-class adapters work **without model weights**: everything
except `recompute` needs only the precomputed embeddings and the metadata.

A new corpus is a `CorpusAdapter` subclass of about a hundred lines — see
[docs/extending.md](docs/extending.md), or run `bioguard new-adapter`.

---

## ⚠️ Honest limits

* Every layer here is **evadable by an adaptive attacker**, and two of them
  have measured blind spots listed above. The tool raises the cost of the
  loud attacks and surfaces the population-level ones; it does not make a
  corpus safe.
* It detects **manipulation of a retrieval corpus**. It does not judge
  whether a sequence or a cell is hazardous and must not be described as
  doing so. Use dedicated biosecurity screening for that question.
* It is not an attack toolkit. The synthetic fixtures in
  `biorag_guard.demo` exist to test defences; no poisoned index, corpus or
  mapping file is produced or committed.

---

## 📚 Documentation

| | |
|---|---|
| [Quickstart](docs/quickstart.md) | four commands from a corpus path to a report |
| [Two modes](docs/two-modes.md) | why ingest and scan are different defences, with the measurements |
| [Layer reference](docs/layers.md) | every layer, what it catches, what it misses |
| [Evaluation](docs/evaluation.md) | detection rates at a stated budget, including the arms where nothing works |
| [Adding a corpus](docs/extending.md) | a new adapter in about a hundred lines |
| [Decisions](docs/decisions/) | the architectural choices, and what was measured to make them |

---

## 📎 Provenance

The cell-corpus layers are ported from
[Single_Cell_Poisoning](https://github.com/Georgakopoulos-Soares-lab/Single_Cell_Poisoning)
(MIT), keeping their names and semantics; `tests/` runs the original
implementations side by side with the ports and asserts they agree. The
text-protein layers are informed by
[prodva_rag_poisoning](https://github.com/Georgakopoulos-Soares-lab/prodva_rag_poisoning).

---

## 🏛️ About

BioRAG-Guard is open research software from
**[BioSafeLabs](https://biosafelabs.org)**.

MIT licensed. Security issues and bypass reports: see
[SECURITY.md](SECURITY.md). Contributions: see
[CONTRIBUTING.md](CONTRIBUTING.md).
