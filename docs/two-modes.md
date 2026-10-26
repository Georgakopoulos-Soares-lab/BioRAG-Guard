# The two modes

## 🎯 Why they are different

They answer different threat models, which the research names and keeps
apart:

* **insert** (`no_access`) &mdash; the attacker cannot write to the
  corpus and can only get records *accepted* into it;
* **overwrite** (`with_access`) &mdash; the attacker edits records that
  are already there.

The defences barely overlap, which is why this is two commands and not
one with a flag.

## 📥 Check at ingest &mdash; `bioguard check`

Runs record-local and corpus-relative layers on an arriving batch, scored
against the corpus as it stood before the batch.

Measured effect against the insert arm, as mean within-celltype attack
success (26 cell types, two doses):

| attack | undefended | `recompute` | `dedup` | `label_consistency` | `donor_cap1` | stack |
|---|---|---|---|---|---|---|
| `duplicate` | 0.343 | 0.343 | 0.001 | 0.002 | 0.010 | 0.001 |
| `relabel` | 0.343 | 0.343 | 0.001 | 0.002 | 0.010 | 0.001 |
| `flood` | 0.264 | 0.001 | 0.145 | 0.004 | 0.023 | 0.001 |
| `interpolate` | 0.005 | 0.001 | 0.005 | 0.002 | 0.003 | 0.001 |

Read the `recompute` and `dedup` columns together. Each is blind exactly
where the other works, and the stack reaches 0.001 on all four attacks
only because both families are present.

**Retrieval does not change.** In the same grid the retrieval rate was
identical across all eight configurations &mdash; 0.810 for
duplicate/relabel, 0.742 for flood, 0.025 for interpolate. These are
vote-time filters; they change the answer, not what is retrieved. The
report keeps the two columns separate for that reason.

## 🔍 Scan a stored corpus &mdash; `bioguard scan`

Adds the layers that need the whole corpus, and the manifest comparison.

Against an attacker who overwrites records, `recompute` still works on
anything that moves a vector &mdash; `shifted_inplace` 0.306 to 0.006,
`synthesised_inplace` 0.112 to 0.004. But against `unmoved_inplace`, the
label rewritten with the embedding untouched:

| configuration | attack success |
|---|---|
| undefended | 0.162 |
| `label_consistency` | 0.149 |
| `dedup` | 0.163 |
| `recompute` | 0.162 |
| full stack | 0.150 |
| `donor_cap1` | **0.178** |

Nothing works, and the donor cap makes it worse. This null is reported
with the same weight as the positive results because it is what justifies
the second mode.

What does work is arithmetic over the label column. Over a 7,913,892-record
reference:

| check | detects a relabel in place | wall time |
|---|---|---|
| hash of the index | ❌ **no** | 1.37 s |
| hash of the label column | ✅ yes | 0.10 s |
| element-wise label diff | ✅ yes | 0.31 s |

## 🗳️ The third surface &mdash; `guard.vote`

`donor_cap`, `weight`, `abstain` and the seasoning window have no
ingest-time form: they change how a retrieved neighbourhood becomes an
answer. They are configured under `vote:` and applied by `guard.vote`,
which returns the defended answer **and** the undefended one, because a
defence that changed nothing and a defence that never ran look identical
from the answer alone.

Two cautions carried over from the measurements. Proximity weighting is
not a filter and may *amplify* a cluster flood, which sits closer to the
query than genuine neighbours do. And an abstention is a refusal, not an
error; report the abstention rate beside the accuracy or the comparison
means nothing.
