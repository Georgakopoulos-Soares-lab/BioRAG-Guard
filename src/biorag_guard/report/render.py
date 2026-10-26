"""Turn a :class:`Report` into something a person reads.

Three rules the renderers enforce rather than trust:

* a layer that did not run is shown, with the plain reason;
* retrieval-side and outcome-side numbers are never merged;
* when a sweep sampled, the sentence saying so appears above the counts,
  and the counts are labelled estimates.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ..core.finding import Report

_VERDICT_COLOUR = {
    "admit": "#1a7f37", "flag": "#9a6700", "quarantine": "#bc4c00", "reject": "#cf222e",
}


def render_terminal(report: Report, width: int = 78) -> str:
    """The transcript form: counts, why, then where to look."""
    lines: list[str] = []
    head = f"{report.command} - {report.adapter} corpus at {report.corpus}"
    lines.append(head)
    lines.append("=" * min(len(head), width))

    for note in report.notes:
        lines.append(_wrap(note, width))
    if report.notes:
        lines.append("")

    if report.results:
        counts = report.counts()
        lines.append("  admit       {:>7,}".format(counts["admit"]))
        lines.append("  flag        {:>7,}".format(counts["flag"]))
        lines.append("  quarantine  {:>7,}".format(counts["quarantine"]))
        lines.append("  reject      {:>7,}".format(counts["reject"]))
        lines.append("")

    if report.findings:
        lines.append("Why:")
        for layer, n in report.by_layer().items():
            sample = next(f for f in report.findings if f.layer == layer)
            lines.append(f"  {layer:<22} {n:>6,}  {_clip(sample.reason, width - 32)}")
        lines.append("")

    if report.calibration:
        lines.extend(_calibration_block(report.calibration, width))

    if report.skipped:
        lines.append("Layers that did not run:")
        for s in report.skipped:
            lines.append(f"  {s.layer:<22} {_clip(s.reason, width - 26)}")
        lines.append("")

    lines.append(f"verdict: {report.verdict.value}   exit code {report.exit_code}")
    return "\n".join(lines)


def _calibration_block(cal: dict[str, Any], width: int) -> list[str]:
    lines = ["Thresholds, set to sacrifice at most {:.2%} of genuine records:".format(
        cal.get("budget", 0.0))]
    for name, lb in cal.get("layers", {}).items():
        lines.append(
            f"  {name:<22} tau={lb['threshold']:<12.6g} "
            f"genuine flagged {lb['flagged_fraction']:.2%}"
        )
    cost = cal.get("cost", {})
    lines.append("")
    if cost.get("available"):
        lines.append("What the vote policy costs on clean data:")
        lines.append(
            f"  clean accuracy    {cost['clean_accuracy']:.3f} -> "
            f"{cost['clean_accuracy_defended']:.3f}  ({cost['accuracy_delta']:+.3f})"
        )
        lines.append(f"  abstentions       {cost['abstention_rate']:.1%}")
        lines.append(f"  policy            {cost['policy']}")
    else:
        lines.append(f"Cost on clean data: not measured - {cost.get('why', 'unavailable')}")
    lines.append("")
    return lines


def render_html(report: Report) -> str:
    """One self-contained file. No CDN, no fonts, no network at open time."""
    d = report.to_dict()
    colour = _VERDICT_COLOUR.get(d["verdict"], "#57606a")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>BioRAG-Guard {html.escape(d['command'])} report</title>",
        "<style>", _CSS, "</style></head><body><main>",
        f"<h1>BioRAG-Guard &mdash; {html.escape(d['command'])}</h1>",
        f"<p class='sub'>{html.escape(d['adapter'])} corpus at "
        f"<code>{html.escape(d['corpus'])}</code> &middot; {html.escape(d['created_at'])}</p>",
        f"<p class='verdict' style='background:{colour}'>verdict: {d['verdict']} "
        f"(exit {d['exit_code']})</p>",
    ]

    if d["notes"]:
        parts.append("<section><h2>Read this first</h2><ul>")
        parts += [f"<li>{html.escape(n)}</li>" for n in d["notes"]]
        parts.append("</ul></section>")

    if any(d["counts"].values()):
        parts.append("<section><h2>Verdicts</h2><table><tr>")
        parts += [f"<th>{k}</th>" for k in d["counts"]]
        parts.append("</tr><tr>")
        parts += [f"<td>{v:,}</td>" for v in d["counts"].values()]
        parts.append("</tr></table></section>")

    if d["retrieval"] or d["outcome"]:
        parts.append("<section><h2>Retrieval and outcome</h2>")
        parts.append("<p class='sub'>Reported separately on purpose: a layer can leave "
                     "retrieval untouched and still change the answer.</p>")
        parts.append(_kv_table({"retrieval": d["retrieval"], "outcome": d["outcome"]}))
        parts.append("</section>")

    if d["calibration"]:
        parts.append("<section><h2>Calibration</h2>")
        parts.append(_calibration_table(d["calibration"]))
        parts.append("</section>")

    if d["findings"]:
        parts.append("<section><h2>Findings</h2>")
        by_layer: dict[str, list[dict[str, Any]]] = {}
        for f in d["findings"]:
            by_layer.setdefault(f["layer"], []).append(f)
        for layer, items in sorted(by_layer.items(), key=lambda kv: -len(kv[1])):
            parts.append(f"<h3>{html.escape(layer)} &middot; {len(items):,}</h3>")
            parts.append("<table><tr><th>record</th><th>score</th><th>verdict</th>"
                         "<th>why</th></tr>")
            for f in items[:200]:
                parts.append(
                    f"<tr><td><code>{html.escape(str(f['record_id']))}</code></td>"
                    f"<td>{f['score']:.4g}</td>"
                    f"<td><span class='pill' style='background:"
                    f"{_VERDICT_COLOUR.get(f['verdict'], '#57606a')}'>{f['verdict']}</span>"
                    f"</td><td>{html.escape(f['reason'])}</td></tr>"
                )
            if len(items) > 200:
                parts.append(f"<tr><td colspan='4'>&hellip; {len(items) - 200:,} more; see "
                             f"report.json</td></tr>")
            parts.append("</table>")
        parts.append("</section>")

    parts.append("<section><h2>Layers that did not run</h2>")
    if d["skipped"]:
        parts.append("<table><tr><th>layer</th><th>reason</th></tr>")
        parts += [
            f"<tr><td><code>{html.escape(s['layer'])}</code></td>"
            f"<td>{html.escape(s['reason'])}</td></tr>" for s in d["skipped"]
        ]
        parts.append("</table>")
    else:
        parts.append("<p>Every enabled layer ran.</p>")
    parts.append("</section>")

    parts.append("<section><h2>Provenance</h2>")
    parts.append(f"<pre>{html.escape(json.dumps(d['provenance'], indent=2))}</pre>")
    parts.append("</section>")
    parts.append("</main></body></html>")
    return "".join(parts)


def _kv_table(data: dict[str, Any]) -> str:
    rows = []
    for group, inner in data.items():
        if not inner:
            continue
        for k, v in inner.items():
            rows.append(f"<tr><td>{html.escape(group)}</td><td>{html.escape(str(k))}</td>"
                        f"<td>{html.escape(str(v))}</td></tr>")
    if not rows:
        return "<p>nothing measured</p>"
    return "<table><tr><th>side</th><th>quantity</th><th>value</th></tr>" + "".join(rows) \
        + "</table>"


def _calibration_table(cal: dict[str, Any]) -> str:
    out = [
        f"<p>Budget {cal.get('budget', 0):.2%} &mdash; the share of genuine records each "
        f"layer may sacrifice. Every threshold below is derived from it.</p>",
        "<table><tr><th>layer</th><th>threshold</th><th>direction</th>"
        "<th>genuine records flagged</th><th>fitted on</th></tr>",
    ]
    for name, lb in cal.get("layers", {}).items():
        out.append(
            f"<tr><td><code>{html.escape(name)}</code></td><td>{lb['threshold']:.6g}</td>"
            f"<td>{lb['direction']}</td><td>{lb['flagged_fraction']:.2%}</td>"
            f"<td>{lb['n_fitted']:,}</td></tr>"
        )
    out.append("</table>")
    cost = cal.get("cost", {})
    if cost.get("available"):
        out.append(
            f"<p>Cost on clean data under policy <code>{html.escape(cost['policy'])}</code>: "
            f"accuracy {cost['clean_accuracy']:.3f} &rarr; "
            f"{cost['clean_accuracy_defended']:.3f} "
            f"({cost['accuracy_delta']:+.3f}), abstentions "
            f"{cost['abstention_rate']:.1%} over {cost['n_probes']:,} probes.</p>"
        )
    else:
        out.append(f"<p>Cost on clean data not measured: "
                   f"{html.escape(str(cost.get('why', 'unavailable')))}.</p>")
    return "".join(out)


def write_report(report: Report, directory: str | Path, formats: list[str]) -> list[Path]:
    """Write the requested formats and return what was written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if "json" in formats:
        p = directory / "report.json"
        p.write_text(report.to_json())
        written.append(p)
    if "html" in formats:
        p = directory / "report.html"
        p.write_text(render_html(report))
        written.append(p)
    return written


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width)) or text


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: max(width - 1, 0)] + "\u2026"


_CSS = """
body{font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  margin:0;background:#f6f8fa;color:#1f2328}
main{max-width:1000px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}
h3{font-size:15px;margin:20px 0 6px;font-family:ui-monospace,monospace}
.sub{color:#57606a;margin:0 0 16px}
.verdict{display:inline-block;color:#fff;padding:4px 12px;border-radius:6px;font-weight:600}
.pill{display:inline-block;color:#fff;padding:1px 8px;border-radius:10px;font-size:12px}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #d0d7de;
  border-radius:6px;overflow:hidden;margin:8px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #eaeef2;vertical-align:top}
th{background:#f6f8fa;font-weight:600;font-size:13px}
tr:last-child td{border-bottom:none}
code{font-family:ui-monospace,monospace;font-size:13px;background:#eff1f3;padding:1px 4px;
  border-radius:4px}
pre{background:#fff;border:1px solid #d0d7de;border-radius:6px;padding:12px;overflow:auto;
  font-size:12px}
section{margin-bottom:8px}
"""
