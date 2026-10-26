# BioRAG-Guard

Point it at a biological retrieval corpus and it answers one question:
**has anything been planted in here, and should I accept this new
record?**

```bash
pip install biorag-guard
bioguard demo
```

## Start here

* [Quickstart](quickstart.md) -- three commands from a corpus path to a
  report.
* [The two jobs](two-jobs.md) -- why guarding at ingest and guarding on
  demand are different defences, with the measurements that make them so.
* [Layer reference](layers.md) -- every layer, what it catches, and what
  it misses. Generated from the registry.
* [Evaluation](evaluation.md) -- detection rates at a stated budget,
  including the arms where nothing works.
* [Adding a corpus](extending.md) -- a new adapter in about a hundred
  lines.

## The one thing to take away

Against an attacker who rewrites the label of a record already in your
corpus, every vote-time defence measured no effect, and a per-donor vote
cap made it worse. What catches it is a hash of the label column, which
over a 7.9-million-record reference costs a tenth of a second -- while a
hash of the index costs fourteen times as much and catches nothing,
because the vectors never move.

So the first command is not a detector:

```bash
bioguard manifest create /data/your_corpus
```
