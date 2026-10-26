# 🧬 BioRAG-Guard

**Integrity checks for biological retrieval corpora.** Screen new records.
Audit existing collections.

A research tool from [BioSafeLabs](https://biosafelabs.org).

```bash
pip install biorag-guard
bioguard demo
```

## 🔬 Why a retrieval corpus needs guarding

A growing class of biological tools answers questions by **looking things
up**. A single-cell model such as
[SCimilarity](https://github.com/Genentech/scimilarity) turns a cell's
expression profile into a vector, then reads the cell type off the nearest
rows of a 7.9-million-cell reference atlas. A text-keyed protein corpus
such as [ProDVa](https://github.com/sornkL/ProDVa) turns a plain-English
description of a function into retrieved sequence fragments that go
straight into a designed protein.

In both cases the corpus *is* the answer. Changing a model means
retraining it; changing a corpus means editing a row — and these
references are assembled from public submissions, shipped as downloadable
tarballs, and copied into labs where nobody re-derives them.

BioRAG-Guard answers one question about such a corpus: **has anything been
planted in here, and should I accept this new record?** It does not judge
whether a sequence or a cell is hazardous, and must not be described as
doing so.

## 📚 Start here

* [Quickstart](quickstart.md) — four commands from a corpus path to a
  report.
* [The two modes](two-modes.md) — why checking at ingest and scanning a
  stored corpus are different defences, with the measurements that make
  them so.
* [Layer reference](layers.md) — every layer, what it catches, and what
  it misses. Generated from the registry.
* [Evaluation](evaluation.md) — detection rates at a stated budget,
  including the arms where nothing works.
* [Adding a corpus](extending.md) — a new adapter in about a hundred
  lines.

## 🧾 The one thing to take away

Against an attacker who rewrites the label of a record already in your
corpus, every vote-time defence measured no effect, and a per-donor vote
cap made it worse. All of those filters ask whether a record's vector
looks out of place, and in this attack the vector is genuine and has not
moved.

What catches it is a hash of the label column, which over a
7.9-million-record reference costs a tenth of a second — while a hash of
the index costs fourteen times as much and catches nothing, because the
vectors never move.

A comparison needs something to compare against, so the first command is
not a detector:

```bash
bioguard manifest create /data/your_corpus
```
