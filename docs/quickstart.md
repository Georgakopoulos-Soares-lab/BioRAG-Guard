# Quickstart

## See it work, with no data

```bash
pip install biorag-guard
bioguard demo
```

Builds a synthetic corpus, takes a manifest, calibrates, screens one batch
per measured attack shape, then relabels records in place and shows that
the index hash misses it while the label hash does not. About a second,
no downloads, no model weights.

## Your own corpus

```bash
bioguard init /data/scimilarity_model_v1.1
```

Detects the layout, reports what the corpus carries, and writes a
commented `bioguard.yaml` you do not have to edit.

```bash
bioguard manifest create /data/scimilarity_model_v1.1
```

Hashes the vectors and the label column **separately**, and keeps the
label column beside the manifest so a later relabel can be named record by
record. Do this first: it is the only thing that catches an in-place edit,
and it is the cheapest check in the tool.

```bash
bioguard calibrate /data/scimilarity_model_v1.1 --budget 0.05
```

Fits every threshold so each layer sacrifices at most 5% of genuine
records, then measures what the policy cost on clean data and prints both.
If the corpus is already poisoned the baseline absorbs the poison; the
command says so, and `--baseline-from` lets you fit on a trusted subset.

```bash
bioguard scan /data/scimilarity_model_v1.1
```

## Screening submissions

```bash
bioguard check new_study/ --corpus /data/scimilarity_model_v1.1
```

Exit code 0 for nothing above a flag, 1 for a quarantine, 2 for a reject,
3 for a run error &mdash; so it drops into CI unchanged.

Screen the batch **before** ingesting it. `label_consistency` scores a
record against the corpus as it stood beforehand; once the batch is
indexed, copies of one poisoned record corroborate each other. The tool
refuses a batch it finds already present rather than reporting a quietly
weakened score.

## In Python

```python
from biorag_guard import Guard, Verdict

guard = Guard.load("bioguard.yaml")

result = guard.check(record)
if result.verdict is Verdict.ADMIT:
    index.add(record)
else:
    print(result.verdict, result.reasons)
```

## What to do with a finding

Every finding carries a `reason` written for a bench scientist and a
`remedy` saying what to do. Start with the report's own grouping: a single
flagged record is usually a submission to query, while a whole study
appearing under one layer is usually a pipeline problem rather than an
attack.
