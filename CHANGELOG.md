# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic
versioning.

## [0.1.0] - 2026-10-26

First release.

### Added

* Two modes: `bioguard check` (check at ingest) and `bioguard scan`
  (scan a stored corpus), split along the insert/overwrite threat models.
* `bioguard manifest create|verify`, hashing the index and the label
  column separately so an in-place relabel is visible when a checksum over
  the vectors is not.
* Cell-corpus layers ported from Single_Cell_Poisoning with their
  semantics preserved: `recompute`, `label_consistency`, `dedup`,
  `donor_cap`, `weight`, `abstain`, `seasoning`.
* Population layers that per-record screening cannot provide: `hub`,
  `duplicate_cluster`, `idx_meta`.
* Text-protein layers: `crossmodal_agreement`, `crossmodal_maps`,
  `text_nearduplicate`, `text_instruction`, `text_fluency`,
  `fragment_outlier`.
* Budget-based calibration: every threshold derived from a measured
  sacrifice rate, with the clean-data cost reported beside it.
* Adapters for SCimilarity, ProDVa and a generic vector corpus, all
  working without model weights except `recompute`.
* `bioguard demo`, the whole flow on a synthetic corpus in about a second.
* `bioguard eval`, replaying the measured attack arms at a fixed budget.
* `bioguard new-adapter`, scaffolding an adapter stub and the test it has
  to pass.
* Service mode (`bioguard serve`) and the `Guard` Python API.
