"""``Guard``: the one object the integration contract is written against.

Two jobs, matching the two threat models:

* :meth:`Guard.check` -- guard at ingest. Record-local and corpus-relative
  layers only, because a hub is invisible until you sweep and an in-place
  edit never passes an ingest gate at all.
* :meth:`Guard.scan` -- guard on demand. Everything, including the
  manifest comparison that catches the attack no vote-time layer touches.

And a third surface, :meth:`Guard.vote`, because ``donor_cap``, ``weight``
and ``abstain`` have no ingest-time form and the measurements put the vote
where the insert-arm attacks are actually defeated.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .adapters import CorpusAdapter, open_corpus, sniff_all
from .calibrate import calibrate_layers, clean_cost, sample_split
from .core.baseline import Baseline
from .core.config import GuardConfig, LayerConfig, default_layers_for
from .core.errors import CorpusAlreadyContainsBatch, MissingEvidence, Quarantined
from .core.finding import Finding, RecordResult, Report, Skipped
from .core.manifest import Manifest
from .core.record import Access, Record, Scope, Verdict
from .core.registry import Context, Layer, get_layer
from .layers import scoring  # noqa: F401  (import registers every layer)
from .vote import VotePolicy, defended_vote, survivors


@dataclass
class VoteResult:
    """One defended answer, with the undefended one beside it.

    Both are reported because a defence that changes nothing and a defence
    that was never applied look identical from the answer alone.
    """

    label: str | None
    undefended_label: str | None
    abstained: bool
    n_neighbours: int
    n_survivors: int
    policy: str

    @property
    def changed(self) -> bool:
        return self.label != self.undefended_label


class Guard:
    """Open a corpus, apply layers, produce reports."""

    def __init__(
        self,
        adapter: CorpusAdapter,
        config: GuardConfig | None = None,
        baseline: Baseline | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or GuardConfig(path=Path(str(adapter.path)),
                                            adapter=adapter.name)
        self.baseline = baseline
        if self.baseline is not None:
            self.baseline.check_against(self.adapter)

    # -- construction -----------------------------------------------------

    @classmethod
    def auto(cls, path: str | Path, **kwargs: Any) -> Guard:
        """Open a corpus by sniffing its layout. No config file needed."""
        adapter = open_corpus(path, **kwargs)
        cfg = GuardConfig.for_corpus(path, adapter.name)
        cfg.layers = {
            name: LayerConfig() for name in default_layers_for(adapter.modality)
        }
        guard = cls(adapter, cfg)
        guard._load_baseline_if_present()
        return guard

    @classmethod
    def load(cls, config_path: str | Path) -> Guard:
        """Open a corpus from a ``bioguard.yaml``."""
        cfg = GuardConfig.load(config_path)
        adapter = open_corpus(cfg.path, adapter=cfg.adapter, **cfg.adapter_options)
        guard = cls(adapter, cfg)
        guard._load_baseline_if_present()
        return guard

    def _load_baseline_if_present(self) -> None:
        path = self.config.baseline_path
        if path.exists():
            self.baseline = Baseline.load(path)
            self.baseline.check_against(self.adapter)

    # -- layer assembly ----------------------------------------------------

    def _enabled_names(self) -> list[str]:
        if self.config.layers:
            return [n for n, c in self.config.layers.items() if c.enabled]
        return default_layers_for(self.adapter.modality)

    def _build(
        self, job: str, require_threshold: bool = True
    ) -> tuple[list[Layer], list[Skipped]]:
        """Instantiate the layers this job can run, and say why the rest cannot.

        ``require_threshold`` is False only during calibration, which is
        what produces the thresholds in the first place: there a budgeted
        layer is built unthresholded so its ``score`` can be fitted.
        """
        allowed: set[Scope] = (
            {Scope.RECORD_LOCAL, Scope.CORPUS_RELATIVE}
            if job == "check"
            else {Scope.RECORD_LOCAL, Scope.CORPUS_RELATIVE, Scope.CORPUS_WIDE}
        )
        built: list[Layer] = []
        skipped: list[Skipped] = []
        caps = self.adapter.capabilities()
        for name in self._enabled_names():
            try:
                cls = get_layer(name)
            except KeyError as exc:
                skipped.append(Skipped(name, str(exc)))
                continue
            spec = cls.spec
            if spec.scope is Scope.VOTE_TIME:
                skipped.append(Skipped(
                    name,
                    "applies at query time only; configure it under `vote:` and call "
                    "guard.vote() -- a sweep cannot run it",
                ))
                continue
            if spec.scope not in allowed:
                skipped.append(Skipped(
                    name,
                    "needs the whole corpus; hub records and in-place edits are only "
                    "visible to `bioguard scan`, so schedule one",
                ))
                continue
            if not (spec.access & set(self.config.access)):
                skipped.append(Skipped(
                    name,
                    f"not evidence against the threat models this deployment declares "
                    f"({', '.join(a.value for a in self.config.access)})",
                ))
                continue
            if job == "check" and Access.INSERT not in spec.access:
                skipped.append(Skipped(
                    name, "applies to records already in the corpus; run `bioguard scan`"
                ))
                continue
            missing = set(spec.needs) - caps
            if missing:
                skipped.append(Skipped(
                    name, f"not applicable: this corpus carries no {', '.join(sorted(missing))}"
                ))
                continue
            lc = self.config.layers.get(name, LayerConfig())
            params = dict(lc.params)
            if name in ("manifest_index", "manifest_label"):
                params.setdefault("path", str(self.config.manifest_path))
            threshold = None
            if spec.budgeted and require_threshold:
                if self.baseline is None or name not in self.baseline.layers:
                    skipped.append(Skipped(
                        name,
                        "no calibrated threshold; run `bioguard calibrate` so its "
                        "threshold comes from a measured budget rather than a guess",
                    ))
                    continue
                threshold = self.baseline.layers[name].threshold
            built.append(cls(
                threshold=threshold,
                verdict=self.config.verdict_for(name, spec.default_verdict),
                **params,
            ))
        return built, skipped

    def _context(self, batch_ids: Sequence[str] | None = None) -> Context:
        return Context(
            adapter=self.adapter,
            thresholds={} if self.baseline is None else {
                k: v.threshold for k, v in self.baseline.layers.items()
            },
            params={n: c.params for n, c in self.config.layers.items()},
            rng=np.random.default_rng(self.config.seed),
            batch_ids=batch_ids,
        )

    def _run_layers(
        self, layers: Sequence[Layer], records: Sequence[Record], ctx: Context,
        report: Report,
    ) -> list[Finding]:
        findings: list[Finding] = []
        for layer in layers:
            t0 = time.perf_counter()
            try:
                found = layer.run(records, ctx)
            except (NotImplementedError, MissingEvidence) as exc:
                report.skipped.append(Skipped(layer.name, str(exc)))
                continue
            findings.extend(found)
            report.provenance.setdefault("timings", {})[layer.name] = round(
                time.perf_counter() - t0, 3
            )
        return findings

    # -- job 1: guard at ingest -----------------------------------------------

    def check(self, records: Record | Iterable[Record]) -> RecordResult | Report:
        """Screen one record or a batch. One record in, one result out."""
        if isinstance(records, Record):
            report = self.check_many([records])
            return report.results[0]
        return self.check_many(list(records))

    def check_many(
        self, records: Sequence[Record], allow_already_ingested: bool = False
    ) -> Report:
        """Screen a batch against the corpus as it stands before the batch."""
        records = list(records)
        report = Report(command="check", corpus=str(self.adapter.path),
                        adapter=self.adapter.name)
        self._stamp(report)
        if not records:
            report.notes.append("no records were submitted")
            return report

        present = set(np.asarray(self.adapter.ids, dtype=object).tolist())
        overlap = [r.id for r in records if r.id in present]
        if overlap and not allow_already_ingested:
            raise CorpusAlreadyContainsBatch(
                f"{len(overlap)} of {len(records)} submitted records are already in the "
                f"corpus (first: {overlap[0]!r}).\n"
                "  label_consistency scores a record against the corpus as it stood "
                "BEFORE the batch; with the batch already indexed, copies of one poisoned "
                "record corroborate each other and the layer silently weakens.\n"
                "  screen the batch before ingesting it, or pass "
                "allow_already_ingested=True and read the scores as a lower bound."
            )

        layers, skipped = self._build("check")
        report.skipped.extend(skipped)
        ctx = self._context(batch_ids=[r.id for r in records])
        findings = self._run_layers(layers, records, ctx, report)
        report.add(findings)

        by_record: dict[str, list[Finding]] = {r.id: [] for r in records}
        for f in findings:
            by_record.setdefault(f.record_id, []).append(f)
        for rec in records:
            fs = by_record.get(rec.id, [])
            verdict = Verdict.ADMIT
            for f in fs:
                verdict = verdict.worst(f.verdict)
            report.results.append(RecordResult(rec.id, verdict, tuple(fs)))

        report.notes.append(
            "Ingest screening cannot see hub records or in-place edits. Schedule "
            "`bioguard scan` for those."
        )
        return report

    # -- job 2: guard on demand ------------------------------------------------

    def scan(
        self, sample: int | None = -1, since: str | datetime | None = None,
    ) -> Report:
        """Sweep the corpus. Sampling is always stated, never silent."""
        report = Report(command="scan", corpus=str(self.adapter.path),
                        adapter=self.adapter.name)
        self._stamp(report)
        n_total = len(self.adapter)
        take = self.config.sample if sample == -1 else sample
        if take is not None and take < n_total:
            rng = np.random.default_rng(self.config.seed)
            pick = np.sort(rng.choice(n_total, size=take, replace=False))
            all_records = list(self.adapter.iter_records())
            records = [all_records[i] for i in pick]
            report.notes.append(
                f"Sampled {take:,} of {n_total:,} records (seed {self.config.seed}). "
                f"Counts below are estimates over that sample, not corpus totals."
            )
        else:
            records = list(self.adapter.iter_records())
        if isinstance(since, str):
            since = datetime.fromisoformat(since)
        if since is not None:
            before = len(records)
            records = [r for r in records if r.added_at and r.added_at >= since]
            report.notes.append(
                f"Restricted to {len(records):,} of {before:,} records added since "
                f"{since.isoformat()}."
            )

        layers, skipped = self._build("scan")
        report.skipped.extend(skipped)
        ctx = self._context()
        report.add(self._run_layers(layers, records, ctx, report))
        report.retrieval = {"n_records": n_total, "n_examined": len(records)}
        report.provenance["sample"] = len(records)
        return report

    # -- job 3: the defended vote ------------------------------------------------

    def vote(self, query: np.ndarray, k: int | None = None) -> VoteResult:
        """Answer one query under the configured vote policy."""
        if self.adapter.vectors is None or self.adapter.labels is None:
            raise MissingEvidence("this corpus has no labels, so there is nothing to vote on")
        vc = self.config.vote
        k = int(k or vc.k)
        policy = self.vote_policy()
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        idx, cos = self.adapter.neighbors(q, k)
        nbr = idx[0]
        labels = np.asarray(self.adapter.labels, dtype=object)[nbr]
        donors = (
            None if self.adapter.donors is None
            else np.asarray(self.adapter.donors, dtype=object)[nbr]
        )
        lc = None
        if policy.label_consistency:
            lc = np.array([
                self._lc_cache().get(str(np.asarray(self.adapter.ids)[j]), 1.0) for j in nbr
            ])
        emb = np.asarray(self.adapter.vectors)[nbr] if policy.dedup_tau else None
        recompute_ok = None
        if policy.recompute:
            raise MissingEvidence(
                "recompute is enabled at vote time but the corpus does not carry the raw "
                "counts of its own records, so no neighbour can be re-derived. Disable it "
                "under `vote:` rather than letting it pass every record."
            )
        kept = survivors(labels, policy, recompute_ok=recompute_ok, label_consistency=lc,
                         donor=donors, emb=emb)
        label = defended_vote(labels, policy, recompute_ok=recompute_ok,
                              label_consistency=lc, donor=donors, cos=cos[0], emb=emb)
        undefended = defended_vote(labels, VotePolicy(), cos=cos[0])
        return VoteResult(
            label=label, undefended_label=undefended, abstained=label is None,
            n_neighbours=int(len(nbr)), n_survivors=int(kept.size), policy=policy.name,
        )

    def vote_policy(self) -> VotePolicy:
        vc = self.config.vote
        lc_threshold = 0.5
        if self.baseline and "label_consistency" in self.baseline.layers:
            lc_threshold = self.baseline.layers["label_consistency"].threshold
        return VotePolicy(
            recompute=vc.recompute,
            label_consistency=vc.label_consistency,
            lc_threshold=lc_threshold,
            donor_cap=vc.donor_cap,
            quarantine=bool(vc.seasoning_days),
            dedup_tau=vc.dedup_tau,
            weight=vc.weight,
            weight_temp=vc.weight_temp,
            abstain=vc.abstain,
            abstain_min_margin=vc.abstain_min_margin,
        )

    def _lc_cache(self) -> dict[str, float]:
        if not hasattr(self, "_lc"):
            scores = scoring.score_records(
                np.asarray(self.adapter.vectors),
                np.asarray(self.adapter.labels, dtype=object),
                np.asarray(self.adapter.vectors),
                np.asarray(self.adapter.labels, dtype=object),
                k=self.config.vote.k,
            )
            self._lc = {
                str(i): float(s)
                for i, s in zip(np.asarray(self.adapter.ids, dtype=object), scores)
            }
        return self._lc

    # -- calibration -------------------------------------------------------------

    def calibrate(
        self, budget: float | None = None, sample: int | None = None, seed: int | None = None,
    ) -> tuple[Baseline, Report]:
        """Fit every budgeted layer to a budget and measure what it costs."""
        budget = float(budget if budget is not None else self.config.budget)
        seed = int(seed if seed is not None else self.config.seed)
        n = sample if sample is not None else (self.config.sample or len(self.adapter))
        report = Report(command="calibrate", corpus=str(self.adapter.path),
                        adapter=self.adapter.name)
        self._stamp(report)

        split = sample_split(self.adapter, n=min(n, len(self.adapter)), seed=seed)
        baseline = Baseline.empty(self.adapter, budget=budget, seed=seed)
        baseline.notes.append(
            "Thresholds are fitted on records assumed genuine. If this corpus is already "
            "poisoned the baseline absorbs the poison; fit from a trusted subset instead."
        )
        layers, skipped = self._build_for_calibration()
        report.skipped.extend(skipped)
        ctx = self._context()
        baseline, could_not = calibrate_layers(layers, split, ctx, budget, baseline)
        for name, why in could_not:
            report.skipped.append(Skipped(name, why))

        self.baseline = baseline
        policy = self.vote_policy()
        baseline.cost = clean_cost(
            self.adapter, split, policy, k=self.config.vote.k,
            lc_scores=self._lc_cache() if policy.label_consistency else None,
        )
        report.calibration = {
            "budget": budget,
            "n_fit": split.sizes[0],
            "n_held_out": split.sizes[1],
            "layers": {k: v.to_dict() for k, v in baseline.layers.items()},
            "cost": baseline.cost,
        }
        report.notes.extend(baseline.notes)
        return baseline, report

    def _build_for_calibration(self) -> tuple[list[Layer], list[Skipped]]:
        """Every layer that can run, unthresholded; only budgeted ones get fitted."""
        return self._build("scan", require_threshold=False)

    # -- manifest ------------------------------------------------------------------

    def manifest_create(self, path: str | Path | None = None) -> Path:
        """Record the corpus state so a later edit becomes visible."""
        target = Path(path or self.config.manifest_path)
        return self.adapter.manifest(keep_labels=True).save(target)

    def manifest_verify(self, path: str | Path | None = None) -> dict[str, Any]:
        """Compare the corpus with a stored manifest."""
        from .core.manifest import compare

        target = Path(path or self.config.manifest_path)
        before = Manifest.load(target)
        after = self.adapter.manifest(keep_labels=True)
        return compare(before, after).to_dict()

    # -- integration helpers ----------------------------------------------------------

    def gate(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator: run an ingest function only for records that pass."""

        @functools.wraps(fn)
        def wrapper(record: Record, *args: Any, **kwargs: Any) -> Any:
            result = self.check(record)
            assert isinstance(result, RecordResult)
            if result.verdict is not Verdict.ADMIT:
                raise Quarantined(record.id, result.verdict.value, result.reasons)
            return fn(record, *args, **kwargs)

        return wrapper

    @contextmanager
    def gating(self, index: Any) -> Iterator[GatedIndex]:
        """Wrap an object with an ``add`` method, buffering what does not pass."""
        gated = GatedIndex(self, index)
        try:
            yield gated
        finally:
            gated.close()

    def quarantine(self, report: Report, records: Sequence[Record]) -> Path | None:
        """Write the held-back records to a reversible side-list."""
        held = {
            r.record_id for r in report.results
            if r.verdict in (Verdict.QUARANTINE, Verdict.REJECT)
        }
        if not held:
            return None
        self.config.quarantine_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = self.config.quarantine_dir / f"{stamp}.jsonl"
        by_id = {r.id: r for r in records}
        reasons = {r.record_id: r.reasons for r in report.results}
        with path.open("w") as fh:
            for rid in sorted(held):
                rec = by_id.get(rid)
                fh.write(json.dumps({
                    "id": rid,
                    "reasons": reasons.get(rid, []),
                    "label": None if rec is None else rec.label,
                    "source": None if rec is None else rec.source,
                }) + "\n")
        return path

    # -- misc --------------------------------------------------------------------------

    def _stamp(self, report: Report) -> None:
        report.provenance.update({
            "biorag_guard": __version__,
            "adapter": self.adapter.describe(),
            "policy": self.config.policy,
            "budget": self.config.budget,
            "access": [a.value for a in self.config.access],
            "seed": self.config.seed,
            "baseline": None if self.baseline is None else {
                "created_at": self.baseline.created_at,
                "budget": self.baseline.budget,
                "n_records": self.baseline.n_records,
            },
        })

    def explain(self, layer_name: str) -> dict[str, Any]:
        """What a layer measures, catches, misses, and is currently set to."""
        cls = get_layer(layer_name)
        spec = cls.spec
        out: dict[str, Any] = {
            "name": spec.name,
            "scope": spec.scope.value,
            "access": sorted(a.value for a in spec.access),
            "needs": sorted(spec.needs),
            "statistic": spec.statistic,
            "catches": spec.catches,
            "misses": spec.misses,
            "ported": spec.ported,
            "budgeted": spec.budgeted,
            "ingest_only": spec.ingest_only,
            "default_verdict": spec.default_verdict.value,
            "default_params": dict(spec.default_params),
        }
        if self.baseline and spec.name in self.baseline.layers:
            lb = self.baseline.layers[spec.name]
            out["calibrated"] = {
                "threshold": lb.threshold,
                "budget": lb.budget,
                "genuine_records_flagged": lb.flagged_fraction,
                "fitted_on": lb.n_fitted,
            }
        else:
            out["calibrated"] = None
        return out


class GatedIndex:
    """An index whose ``add`` only sees records that passed."""

    def __init__(self, guard: Guard, index: Any) -> None:
        self.guard = guard
        self.index = index
        self.rejected: list[tuple[Record, RecordResult]] = []

    def add(self, record: Record, *args: Any, **kwargs: Any) -> Any:
        result = self.guard.check(record)
        assert isinstance(result, RecordResult)
        if result.verdict is not Verdict.ADMIT:
            self.rejected.append((record, result))
            return None
        return self.index.add(record, *args, **kwargs)

    def close(self) -> None:
        pass

    def __getattr__(self, item: str) -> Any:
        return getattr(self.index, item)


def available_adapters(path: str | Path) -> list[tuple[str, float]]:
    """What each adapter makes of this path. Used by ``bioguard init``."""
    return sniff_all(path)
