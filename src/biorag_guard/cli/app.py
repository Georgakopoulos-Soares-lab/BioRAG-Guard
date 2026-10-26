"""``bioguard`` -- four commands get a user from a corpus path to a report.

Exit codes are part of the contract, so the tool drops into CI unchanged:
0 nothing above flag, 1 a quarantine, 2 a reject, 3 a run error.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import typer

from .. import __version__
from .. import demo as demo_mod
from ..adapters import open_corpus, sniff_all
from ..core.arrays import Array
from ..core.config import GuardConfig, LayerConfig, default_layers_for, render_yaml
from ..core.errors import BioGuardError
from ..core.record import EXIT_RUN_ERROR, Record
from ..core.registry import all_layers
from ..guard import Guard
from ..report import render_terminal, write_report

app = typer.Typer(
    add_completion=False,
    help="Integrity checks for biological retrieval corpora. "
         "Check at ingest, scan a stored corpus.",
    no_args_is_help=True,
)


def _echo(msg: str = "") -> None:
    typer.echo(msg)


def _guard(corpus: Path, config: Path | None) -> Guard:
    if config is not None:
        return Guard.load(config)
    local = corpus / "bioguard.yaml"
    if local.exists():
        return Guard.load(local)
    return Guard.auto(corpus)


@app.command()
def version() -> None:
    """Print the version."""
    _echo(f"biorag-guard {__version__}")


@app.command()
def init(
    corpus: Path = typer.Argument(..., help="the corpus directory"),
    out: Path | None = typer.Option(None, help="where to write bioguard.yaml"),
) -> None:
    """Detect the corpus type and write a working config."""
    _echo(f"Looking at {corpus} ...")
    ranked = sniff_all(corpus)
    for name, score in ranked:
        if score > 0:
            _echo(f"  {name:<14} confidence {score:.2f}")
    if not ranked or ranked[0][1] <= 0:
        _echo("  nothing recognised.")
        raise typer.Exit(EXIT_RUN_ERROR)
    adapter = open_corpus(corpus)
    d = adapter.describe()
    _echo(f"  {d['n_records']:,} records, {d['dims']} dimensions, "
          f"carries: {', '.join(d['capabilities'])}")
    _echo(f"Corpus type: {d['adapter']}  (confidence: "
          f"{'high' if ranked[0][1] >= 0.8 else 'medium'})")
    cfg = GuardConfig(path=corpus, adapter=d["adapter"])
    cfg.layers = {n: LayerConfig() for n in default_layers_for(d["modality"])}
    target = out or (corpus / "bioguard.yaml")
    target.write_text(render_yaml(cfg))
    _echo("")
    _echo(f"Wrote {target}. Next:")
    _echo(f"  bioguard manifest create {corpus}   # record state, so edits become visible")
    _echo(f"  bioguard calibrate {corpus}         # learn what normal looks like here")
    _echo(f"  bioguard scan {corpus}              # then sweep the corpus")


@app.command()
def calibrate(
    corpus: Path = typer.Argument(...),
    budget: float = typer.Option(0.05, help="fraction of genuine records layers may lose"),
    sample: int | None = typer.Option(None, help="records to fit on"),
    config: Path | None = typer.Option(None),
    out: Path | None = typer.Option(None, help="where to write baseline.json"),
) -> None:
    """Fit every threshold to a budget and measure what it costs."""
    guard = _guard(corpus, config)
    baseline, report = guard.calibrate(budget=budget, sample=sample)
    target = out or guard.config.baseline_path
    baseline.save(target)
    _echo(render_terminal(report))
    _echo(f"Wrote {target}.")


@app.command()
def check(
    batch: Path = typer.Argument(..., help="a directory or .jsonl of records to screen"),
    corpus: Path = typer.Option(..., help="the corpus to screen against"),
    config: Path | None = typer.Option(None),
    quarantine: bool = typer.Option(True, help="write held-back records to a side-list"),
    json_out: bool = typer.Option(False, "--json", help="print the report as JSON"),
) -> None:
    """Check at ingest: decide on records arriving at the corpus."""
    guard = _guard(corpus, config)
    records = load_records(batch)
    report = guard.check_many(records)
    if json_out:
        _echo(report.to_json())
    else:
        _echo(render_terminal(report))
        if quarantine:
            path = guard.quarantine(report, records)
            if path:
                _echo(f"Held back -> {path}   (bioguard quarantine release <id>)")
        written = write_report(report, guard.config.report_dir, guard.config.report_formats)
        if written:
            _echo("Report: " + "  ".join(str(p) for p in written))
    raise typer.Exit(report.exit_code)


@app.command()
def scan(
    corpus: Path = typer.Argument(...),
    sample: int | None = typer.Option(None, help="records to examine; omit for all"),
    since: str | None = typer.Option(None, help="only records added on or after this date"),
    config: Path | None = typer.Option(None),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Scan a stored corpus: sweep everything already in it."""
    guard = _guard(corpus, config)
    report = guard.scan(sample=sample if sample is not None else -1, since=since)
    if json_out:
        _echo(report.to_json())
    else:
        _echo(render_terminal(report))
        written = write_report(report, guard.config.report_dir, guard.config.report_formats)
        if written:
            _echo("Report: " + "  ".join(str(p) for p in written))
    raise typer.Exit(report.exit_code)


manifest_app = typer.Typer(help="Record corpus state, so a later edit becomes visible.")
app.add_typer(manifest_app, name="manifest")


@manifest_app.command("create")
def manifest_create(
    corpus: Path = typer.Argument(...),
    config: Path | None = typer.Option(None),
    out: Path | None = typer.Option(None),
) -> None:
    """Hash the index and the labels separately, and keep the label column."""
    guard = _guard(corpus, config)
    path = guard.manifest_create(out)
    _echo(f"Wrote {path} over {len(guard.adapter):,} records.")
    _echo("The label column is stored beside it, so a later relabel can be named "
          "record by record.")


@manifest_app.command("verify")
def manifest_verify(
    corpus: Path = typer.Argument(...),
    config: Path | None = typer.Option(None),
    manifest: Path | None = typer.Option(None),
) -> None:
    """Compare the corpus with a stored manifest."""
    guard = _guard(corpus, config)
    diff = guard.manifest_verify(manifest)
    _echo(f"index hash      {'CHANGED' if diff['index_changed'] else 'unchanged'}")
    _echo(f"label hash      {'CHANGED' if diff['labels_changed'] else 'unchanged'}")
    _echo(f"record ids      {'CHANGED' if diff['ids_changed'] else 'unchanged'}")
    if diff["n_labels_changed"]:
        _echo(f"\n{diff['n_labels_changed']:,} records changed label while keeping their "
              f"embedding.")
        for lab, n in list(diff["destination_tally"].items())[:5]:
            _echo(f"   {n:>7,} now labelled {lab!r}")
    if diff["relabel_in_place"]:
        _echo("\nThe index is byte-identical, so a checksum over the vectors would have "
              "reported this corpus as clean.")
    for note in diff["notes"]:
        _echo(f"note: {note}")
    raise typer.Exit(0 if diff["n_before"] and not (
        diff["index_changed"] or diff["labels_changed"] or diff["ids_changed"]) else 2)


@app.command()
def explain(
    layer: str | None = typer.Argument(None, help="layer name; omit to list them all"),
    corpus: Path | None = typer.Option(None, help="show the calibrated threshold too"),
    config: Path | None = typer.Option(None),
) -> None:
    """What a layer measures, what it catches, and what it misses."""
    if layer is None:
        _echo(f"{'layer':<24}{'scope':<18}{'ported':<8}catches")
        _echo("-" * 100)
        for name, cls in all_layers().items():
            _echo(f"{name:<24}{cls.spec.scope.value:<18}"
                  f"{'yes' if cls.spec.ported else 'no':<8}{cls.spec.catches[:52]}")
        return
    guard = _guard(corpus, config) if corpus else None
    info = guard.explain(layer) if guard else _explain_static(layer)
    _echo(f"{info['name']}  ({info['scope']}, "
          f"{'ported from the research code' if info['ported'] else 'new in this tool'})")
    _echo("")
    _echo("measures   " + info["statistic"])
    _echo("catches    " + info["catches"])
    _echo("misses     " + info["misses"])
    _echo("threat     " + ", ".join(info["access"]))
    if info["needs"]:
        _echo("needs      " + ", ".join(info["needs"]))
    if info["ingest_only"]:
        _echo("note       sound at ingest only; see the layer reference")
    cal = info.get("calibrated")
    if cal:
        _echo("")
        _echo(f"on this corpus  threshold {cal['threshold']:.6g} at budget "
              f"{cal['budget']:.2%}, flagging {cal['genuine_records_flagged']:.2%} of "
              f"{cal['fitted_on']:,} genuine records")
    elif corpus:
        _echo("\non this corpus  not calibrated; run `bioguard calibrate`")


def _explain_static(layer: str) -> dict[str, Any]:
    spec = all_layers()[layer].spec
    return {
        "name": spec.name, "scope": spec.scope.value,
        "access": sorted(a.value for a in spec.access), "needs": sorted(spec.needs),
        "statistic": spec.statistic, "catches": spec.catches, "misses": spec.misses,
        "ported": spec.ported, "budgeted": spec.budgeted,
        "ingest_only": spec.ingest_only,
        "default_verdict": spec.default_verdict.value,
        "default_params": dict(spec.default_params), "calibrated": None,
    }


@app.command("eval")
def evaluate_cmd(
    corpus: Path | None = typer.Option(None, help="corpus to evaluate on; omit for a "
                                                  "synthetic one"),
    budget: float = typer.Option(0.05, help="budget the detection rates are measured at"),
    n: int = typer.Option(30, help="poison records per arm"),
    out: Path | None = typer.Option(None, help="write the markdown table here"),
) -> None:
    """Replay the measured attack arms and report detection at a fixed budget."""
    import tempfile

    from ..evaluate import evaluate_insert_arms, render_table

    if corpus is None:
        base = Path(tempfile.mkdtemp(prefix="bioguard-eval-"))
        built = demo_mod.build(base / "corpus")
        guard = Guard.auto(built.path)
        _echo(f"No corpus given; evaluating on a synthetic one at {built.path}.")
    else:
        guard = _guard(corpus, None)
        built = demo_mod.DemoCorpus(
            path=Path(corpus),
            vectors=np.asarray(guard.adapter.vectors),
            rows=[{"id": str(i), "label": str(lbl), "counts": []}
                  for i, lbl in enumerate(guard.adapter.labels or [])],
        )
    results = evaluate_insert_arms(guard, built, n=n, budget=budget)
    table = render_table(results)
    _echo("")
    _echo(table)
    if out:
        out.write_text(table + "\n")
        _echo(f"\nWrote {out}.")


@app.command()
def serve(
    corpus: Path = typer.Argument(...),
    host: str = typer.Option("127.0.0.1", help="bind address"),
    port: int = typer.Option(8713),
    token: str | None = typer.Option(None, help="require this value in X-Bioguard-Token"),
    config: Path | None = typer.Option(None),
) -> None:
    """Serve the same two modes over HTTP, for non-Python ingest pipelines."""
    import uvicorn

    from ..service import create_app

    guard = _guard(corpus, config)
    if host not in ("127.0.0.1", "localhost") and token is None:
        _echo("WARNING: bound to a non-local address with no token. Anyone who can reach "
              "this port can read your corpus layout and submit records for screening. "
              "Pass --token.")
    uvicorn.run(create_app(guard, token=token), host=host, port=port)


@app.command("new-adapter")
def new_adapter(
    name: str = typer.Argument(..., help="corpus name, e.g. my-atlas"),
    out: Path = typer.Option(Path("."), help="directory to write the stub and its test into"),
) -> None:
    """Scaffold an adapter for a new corpus, and the test it has to pass."""
    from ..scaffold import scaffold_adapter

    for path in scaffold_adapter(name, out):
        _echo(f"wrote {path}")
    _echo("\nImplement open()/ids, then build_fixture() in the test. "
          "See docs/extending.md.")


@app.command()
def demo(
    workdir: Path | None = typer.Option(None, help="where to build the demo corpus"),
    keep: bool = typer.Option(False, help="leave the corpus on disk afterwards"),
) -> None:
    """Run the whole flow on a synthetic corpus. No downloads, no weights."""
    import tempfile

    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="bioguard-demo-"))
    corpus_dir = base / "corpus"
    _echo(f"1/5  building a synthetic corpus in {corpus_dir}")
    corpus = demo_mod.build(corpus_dir)
    guard = Guard.auto(corpus_dir)
    _echo(f"     {len(guard.adapter):,} records, "
          f"{guard.adapter.describe()['dims']} dimensions, 4 cell types, 6 donors")

    _echo("\n2/5  taking a manifest")
    guard.manifest_create(base / "manifest.json")
    _echo("     index hash and label hash stored separately, on purpose")

    _echo("\n3/5  calibrating to a 5% budget")
    baseline, cal = guard.calibrate(budget=0.05)
    baseline.save(base / "baseline.json")
    for name, lb in baseline.layers.items():
        _echo(f"     {name:<20} tau={lb.threshold:<10.5g} "
              f"flags {lb.flagged_fraction:.1%} of genuine records")
    cost = baseline.cost
    if cost.get("available"):
        _echo(f"     clean accuracy {cost['clean_accuracy']:.3f} -> "
              f"{cost['clean_accuracy_defended']:.3f}, abstentions "
              f"{cost['abstention_rate']:.1%}")

    _echo("\n4/5  checking at ingest, one batch per measured attack shape")
    _echo(f"     {'attack':<14}{'admitted':>9}{'held':>7}   layers that fired")
    for kind in ("relabel", "duplicate", "interpolate", "flood", "hub"):
        vecs, rows = demo_mod.poison(corpus, kind, n=20)
        records = _records_from(vecs, rows)
        rep = guard.check_many(records)
        c = rep.counts()
        fired = ", ".join(f"{k} x{v}" for k, v in rep.by_layer().items()) or "none"
        _echo(f"     {kind:<14}{c['admit']:>9}{c['quarantine'] + c['reject']:>7}   {fired}")
    _echo("     note: `recompute` is silent on relabel and duplicate and decisive on the")
    _echo("     synthesised ones; `dedup` is the mirror image. Enable one family only and")
    _echo("     you have a hole.")

    _echo("\n5/5  scanning a stored corpus: relabelling records already in it")
    changed = demo_mod.relabel_in_place(corpus, n=30)
    guard2 = Guard.auto(corpus_dir)
    diff = guard2.manifest_verify(base / "manifest.json")
    _echo(f"     index hash   {'CHANGED' if diff['index_changed'] else 'unchanged'}"
          f"   <- a checksum over the vectors sees nothing")
    _echo(f"     label hash   {'CHANGED' if diff['labels_changed'] else 'unchanged'}"
          f"   <- {diff['n_labels_changed']} records named, of {len(changed)} edited")
    _echo("")
    _echo("That asymmetry is why `bioguard scan` exists: against an in-place edit every")
    _echo("vote-time layer measured no effect, and the label manifest is what catches it.")
    if keep:
        _echo(f"\nCorpus left at {base}")


def _records_from(vectors: Array, rows: list[dict[str, Any]]) -> list[Record]:
    out = []
    for i, r in enumerate(rows):
        counts = r.get("counts")
        added = r.get("added_at")
        out.append(Record(
            id=r["id"], vector=vectors[i], label=r.get("label"), donor=r.get("donor"),
            counts=None if counts is None else np.asarray(counts, dtype=np.float32),
            added_at=None if added is None else datetime.fromisoformat(added),
            text=r.get("text"), payload=r.get("payload"),
            features=r.get("features", {}), source=r.get("source"),
        ))
    return out


def load_records(path: Path) -> list[Record]:
    """Read a batch from a generic-layout directory or a .jsonl file."""
    path = Path(path)
    if path.is_dir():
        adapter = open_corpus(path, adapter="generic")
        return list(adapter.iter_records())
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    vecs = np.vstack([
        np.asarray(r["vector"], dtype=np.float32) for r in rows
    ]) if rows and "vector" in rows[0] else np.zeros((len(rows), 0), dtype=np.float32)
    return _records_from(vecs, rows)


def main() -> None:
    try:
        app()
    except BioGuardError as exc:
        typer.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_RUN_ERROR)


if __name__ == "__main__":  # pragma: no cover
    main()
