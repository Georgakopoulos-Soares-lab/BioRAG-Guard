# Layer reference

Generated from the registry by `python -m biorag_guard.docs_gen`. Every field
below is data on the layer, which is also what `bioguard explain <layer>` prints.

**Ported** means the implementation comes from the research code and keeps its
semantics; `tests/` runs the originals beside the ports and asserts they agree.

| layer | scope | ported | catches | misses |
|---|---|---|---|---|
| [`abstain`](#abstain) | vote-time | yes | Converts thin-margin wrong answers into refusals. | Not a detector. A confident poisoned majority is answered confidently. |
| [`crossmodal_agreement`](#crossmodal-agreement) | record-local | no | A plausible description bonded to a payload that is annotated as something else -- the shape the random-control arm isolates. | A payload whose annotation genuinely matches the description, and any record whose annotation terms the corpus does not carry. |
| [`crossmodal_maps`](#crossmodal-maps) | corpus-wide | no | Descriptions indexed but absent from the sequence map, sequences absent from the fragment map, orphaned fragment entries, and colliding description keys that silently overwrite a real record. | Poison that is internally consistent across all three files. |
| [`dedup`](#dedup) | corpus-relative | yes | Verbatim and near-verbatim copies of records already in the corpus, which is how a budgeted attacker concentrates influence on one neighbourhood. | A cluster flood of distinct synthesised records. Unsound at query time -- see the class docstring. |
| [`donor_cap`](#donor-cap) | vote-time | yes | Influence concentrated in copies: many records from one source crowding a single neighbourhood. | Fabricated donor identities, which cost an attacker nothing. Measured as actively harmful against in-place relabelling. |
| [`duplicate_cluster`](#duplicate-cluster) | corpus-wide | no | Many near-identical records already sitting in the corpus, including ones that predate any ingest screening. | A single copy, which `dedup` catches at ingest, and distinct synthesised records, which form no cluster. |
| [`fragment_outlier`](#fragment-outlier) | record-local | no | Abnormally long or numerous fragments entering the dynamic vocabulary. | A payload with ordinary fragment statistics, which is easy to arrange. |
| [`hub`](#hub) | corpus-wide | no | Records positioned to be retrieved for many unrelated queries at once. | Targeted poison aimed at a single query, which by construction is returned rarely. Not detectable at ingest at all -- only a sweep can see it. |
| [`idx_meta`](#idx-meta) | corpus-wide | no | Index/metadata row-count mismatch, duplicate primary keys, orphaned sidecar files. | Metadata that is internally consistent and wrong. |
| [`label_consistency`](#label-consistency) | corpus-relative | yes | The realistic insert attack: a genuine embedding submitted under a label its neighbours do not share. | A label rewritten on a record already in the corpus -- there is no submission to screen. Measured at 0.149 against 0.162 undefended, which is no effect. Use the manifest for that case. |
| [`manifest_index`](#manifest-index) | corpus-wide | yes | Any edit that changes a stored vector: shifted or re-synthesised embeddings, and records inserted into the middle of the index. | A relabel. The label can be rewritten with the index byte-identical, and this check will pass. That is measured, not hypothetical. |
| [`manifest_label`](#manifest-label) | corpus-wide | yes | Labels rewritten in place -- the one attack on which every vote-time layer measured no effect. Names the records and where they moved to. | Edits made before the first manifest was taken. The manifest is a starting point, not a history. |
| [`qc_align`](#qc-align) | record-local | no | Wrong embedding dimension, non-finite values, and vectors that are not unit-norm when the corpus is. | Anything correctly prepared. This is a correctness gate, not a defence. |
| [`qc_counts`](#qc-counts) | record-local | no | Fabricated profiles: impossible library sizes, implausible detection rates, non-integer raw counts. | Poison built from real profiles, which is most of the measured attack taxonomy. This is a hygiene check, not a defence. |
| [`recompute`](#recompute) | record-local | yes | Embeddings that no real measurement could have produced: synthesised geometry, interpolated vectors, and vectors written straight into the index. | Poison built from genuine embeddings. A relabelled or duplicated real cell re-derives perfectly, so this layer passes it. |
| [`seasoning`](#seasoning) | vote-time | yes | Freshly inserted records influencing answers before anyone has reviewed them. | Anything already past the window, and every in-place edit. |
| [`text_fluency`](#text-fluency) | corpus-relative | no | Token-salad and optimised hub descriptions built from words the corpus rarely uses. | Fluent paraphrases, and optimised text that reuses common vocabulary. A unigram surrogate, not a language model. |
| [`text_instruction`](#text-instruction) | record-local | no | Imperatives aimed at the reader, URLs and contact strings inside what should be a description of a protein. | Poison that reads entirely like biology, which is the competent version of this attack. |
| [`text_nearduplicate`](#text-nearduplicate) | corpus-relative | no | Descriptions copied near-verbatim from an existing record or an expected query -- the arm measured at 0.58 capture templated, 0.97 free-text. | A genuine paraphrase, which captures at 0.21 and shares few exact tokens. This layer is not evidence that a corpus is clean. |
| [`weight`](#weight) | vote-time | yes | Nothing on its own. Measured only as a modifier on a filtered neighbourhood. | Everything, considered as a detector. Can make a cluster flood stronger. |

## abstain

*applies when a query is answered; configure under `vote:`*

**Measures.** top-label vote share minus runner-up share.

**Catches.** Converts thin-margin wrong answers into refusals.

**Misses.** Not a detector. A confident poisoned majority is answered confidently.

- threat model: insert, overwrite
- needs: nothing beyond the record
- threshold: fixed
- default verdict: `flag`
- parameters: `min_margin=0.0`

## crossmodal_agreement

*needs only the record; runs at ingest and in a sweep*

**Measures.** agreement between the description and the payload's annotation terms, via the configured backend.

**Catches.** A plausible description bonded to a payload that is annotated as something else -- the shape the random-control arm isolates.

**Misses.** A payload whose annotation genuinely matches the description, and any record whose annotation terms the corpus does not carry.

- threat model: insert, overwrite
- needs: payload, text
- threshold: from the budget, via `bioguard calibrate`
- default verdict: `quarantine`
- parameters: `backend='term_overlap'`

## crossmodal_maps

*needs the whole corpus; **a sweep only** -- ingest screening cannot see this*

**Measures.** set algebra over the index, the sequence map and the fragment map.

**Catches.** Descriptions indexed but absent from the sequence map, sequences absent from the fragment map, orphaned fragment entries, and colliding description keys that silently overwrite a real record.

**Misses.** Poison that is internally consistent across all three files.

- threat model: insert, overwrite
- needs: nothing beyond the record
- threshold: fixed
- default verdict: `reject`

## dedup

*needs one neighbour query against the corpus; runs at ingest and in a sweep*

**Measures.** highest cosine from the arriving record to any record already present.

**Catches.** Verbatim and near-verbatim copies of records already in the corpus, which is how a budgeted attacker concentrates influence on one neighbourhood.

**Misses.** A cluster flood of distinct synthesised records. Unsound at query time -- see the class docstring.

- threat model: insert
- needs: vector
- threshold: from the budget, via `bioguard calibrate`
- default verdict: `quarantine`
- **sound at ingest only** -- at query time an exact copy ties with its original and the tie-break decides which survives
- parameters: `tau=0.999`

## donor_cap

*applies when a query is answered; configure under `vote:`*

**Measures.** votes retained per donor within one query's neighbourhood.

**Catches.** Influence concentrated in copies: many records from one source crowding a single neighbourhood.

**Misses.** Fabricated donor identities, which cost an attacker nothing. Measured as actively harmful against in-place relabelling.

- threat model: insert
- needs: donor
- threshold: fixed
- default verdict: `flag`
- parameters: `cap=1`

## duplicate_cluster

*needs the whole corpus; **a sweep only** -- ingest screening cannot see this*

**Measures.** connected components of the cosine>tau graph, reported at size>=min_size.

**Catches.** Many near-identical records already sitting in the corpus, including ones that predate any ingest screening.

**Misses.** A single copy, which `dedup` catches at ingest, and distinct synthesised records, which form no cluster.

- threat model: insert
- needs: vector
- threshold: fixed
- default verdict: `flag`
- parameters: `tau=0.999`, `min_size=5`, `block=2048`

## fragment_outlier

*needs only the record; runs at ingest and in a sweep*

**Measures.** fragment count and longest fragment, as a robust z against the corpus.

**Catches.** Abnormally long or numerous fragments entering the dynamic vocabulary.

**Misses.** A payload with ordinary fragment statistics, which is easy to arrange.

- threat model: insert
- needs: payload
- threshold: from the budget, via `bioguard calibrate`
- default verdict: `flag`

## hub

*needs the whole corpus; **a sweep only** -- ingest screening cannot see this*

**Measures.** k-occurrence: how many of the probe queries return this record.

**Catches.** Records positioned to be retrieved for many unrelated queries at once.

**Misses.** Targeted poison aimed at a single query, which by construction is returned rarely. Not detectable at ingest at all -- only a sweep can see it.

- threat model: insert, overwrite
- needs: vector
- threshold: from the budget, via `bioguard calibrate`
- default verdict: `quarantine`
- parameters: `probes=5000`, `k=50`, `seed=0`

## idx_meta

*needs the whole corpus; **a sweep only** -- ingest screening cannot see this*

**Measures.** set algebra and row counts across the corpus files.

**Catches.** Index/metadata row-count mismatch, duplicate primary keys, orphaned sidecar files.

**Misses.** Metadata that is internally consistent and wrong.

- threat model: insert, overwrite
- needs: nothing beyond the record
- threshold: fixed
- default verdict: `reject`

## label_consistency

*needs one neighbour query against the corpus; runs at ingest and in a sweep*

**Measures.** fraction of the record's k nearest reference neighbours carrying the same label, self-matches excluded.

**Catches.** The realistic insert attack: a genuine embedding submitted under a label its neighbours do not share.

**Misses.** A label rewritten on a record already in the corpus -- there is no submission to screen. Measured at 0.149 against 0.162 undefended, which is no effect. Use the manifest for that case.

- threat model: insert
- needs: label, vector
- threshold: from the budget, via `bioguard calibrate`
- default verdict: `quarantine`
- parameters: `k=50`

## manifest_index

*needs the whole corpus; **a sweep only** -- ingest screening cannot see this*

**Measures.** blake2b over the vector matrix, plus one digest per 100,000-record block.

**Catches.** Any edit that changes a stored vector: shifted or re-synthesised embeddings, and records inserted into the middle of the index.

**Misses.** A relabel. The label can be rewritten with the index byte-identical, and this check will pass. That is measured, not hypothetical.

- threat model: overwrite
- needs: nothing beyond the record
- threshold: fixed
- default verdict: `reject`
- parameters: `path='bioguard_manifest.json'`

## manifest_label

*needs the whole corpus; **a sweep only** -- ingest screening cannot see this*

**Measures.** blake2b over the label column, then an id-joined element-wise diff.

**Catches.** Labels rewritten in place -- the one attack on which every vote-time layer measured no effect. Names the records and where they moved to.

**Misses.** Edits made before the first manifest was taken. The manifest is a starting point, not a history.

- threat model: overwrite
- needs: label
- threshold: fixed
- default verdict: `reject`
- parameters: `path='bioguard_manifest.json'`

## qc_align

*needs only the record; runs at ingest and in a sweep*

**Measures.** dimension match, finiteness, and norm against the corpus convention.

**Catches.** Wrong embedding dimension, non-finite values, and vectors that are not unit-norm when the corpus is.

**Misses.** Anything correctly prepared. This is a correctness gate, not a defence.

- threat model: insert
- needs: vector
- threshold: fixed
- default verdict: `reject`
- parameters: `norm_tolerance=0.001`

## qc_counts

*needs only the record; runs at ingest and in a sweep*

**Measures.** largest robust z-score (median / MAD) across the count statistics.

**Catches.** Fabricated profiles: impossible library sizes, implausible detection rates, non-integer raw counts.

**Misses.** Poison built from real profiles, which is most of the measured attack taxonomy. This is a hygiene check, not a defence.

- threat model: insert
- needs: counts
- threshold: from the budget, via `bioguard calibrate`
- default verdict: `flag`
- parameters: `stats=('library_size', 'detected_fraction')`

## recompute

*needs only the record; runs at ingest and in a sweep*

**Measures.** cosine distance between the stored vector and the vector re-derived from the record's own counts.

**Catches.** Embeddings that no real measurement could have produced: synthesised geometry, interpolated vectors, and vectors written straight into the index.

**Misses.** Poison built from genuine embeddings. A relabelled or duplicated real cell re-derives perfectly, so this layer passes it.

- threat model: insert, overwrite
- needs: counts, vector
- threshold: from the budget, via `bioguard calibrate`
- default verdict: `quarantine`
- parameters: `tolerance=0.0001`

## seasoning

*applies when a query is answered; configure under `vote:`*

**Measures.** age of the record against the quarantine window.

**Catches.** Freshly inserted records influencing answers before anyone has reviewed them.

**Misses.** Anything already past the window, and every in-place edit.

- threat model: insert
- needs: added_at
- threshold: fixed
- default verdict: `flag`
- parameters: `days=30`

## text_fluency

*needs one neighbour query against the corpus; runs at ingest and in a sweep*

**Measures.** mean unigram surprisal of the description under the corpus vocabulary.

**Catches.** Token-salad and optimised hub descriptions built from words the corpus rarely uses.

**Misses.** Fluent paraphrases, and optimised text that reuses common vocabulary. A unigram surrogate, not a language model.

- threat model: insert
- needs: text
- threshold: from the budget, via `bioguard calibrate`
- default verdict: `flag`

## text_instruction

*needs only the record; runs at ingest and in a sweep*

**Measures.** count of instruction-shaped and contact-shaped phrases.

**Catches.** Imperatives aimed at the reader, URLs and contact strings inside what should be a description of a protein.

**Misses.** Poison that reads entirely like biology, which is the competent version of this attack.

- threat model: insert, overwrite
- needs: text
- threshold: fixed
- default verdict: `quarantine`
- parameters: `threshold=0.5`

## text_nearduplicate

*needs one neighbour query against the corpus; runs at ingest and in a sweep*

**Measures.** highest token Jaccard against the descriptions already in the corpus.

**Catches.** Descriptions copied near-verbatim from an existing record or an expected query -- the arm measured at 0.58 capture templated, 0.97 free-text.

**Misses.** A genuine paraphrase, which captures at 0.21 and shares few exact tokens. This layer is not evidence that a corpus is clean.

- threat model: insert
- needs: text
- threshold: from the budget, via `bioguard calibrate`
- default verdict: `quarantine`
- parameters: `candidates=200`

## weight

*applies when a query is answered; configure under `vote:`*

**Measures.** uniform | score (1/(1-cos)) | cosine | rank | softmax.

**Catches.** Nothing on its own. Measured only as a modifier on a filtered neighbourhood.

**Misses.** Everything, considered as a detector. Can make a cluster flood stronger.

- threat model: insert, overwrite
- needs: nothing beyond the record
- threshold: fixed
- default verdict: `flag`
- parameters: `scheme='uniform'`, `temp=0.01`
