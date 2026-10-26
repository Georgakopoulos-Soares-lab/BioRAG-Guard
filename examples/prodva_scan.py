"""Sweep a description-keyed protein corpus.

The three files that make up the corpus are all attacker-writable, so the
set algebra across them runs first and costs nothing.
"""

from pathlib import Path

from biorag_guard import Guard

CORPUS = Path("/data/prodva_corpus")


def main() -> None:
    guard = Guard.auto(CORPUS)
    if guard.baseline is None:
        guard.calibrate(budget=0.05)

    report = guard.scan(sample=None)

    print(f"verdict: {report.verdict.value}")
    for layer, n in report.by_layer().items():
        print(f"  {layer:<24} {n:>6}")

    for skipped in report.skipped:
        print(f"  (not run) {skipped.layer}: {skipped.reason}")

    # The paraphrased arm is the one that evades text_nearduplicate by
    # construction. crossmodal_agreement is what stands in front of it,
    # and it needs an annotation source -- see docs/decisions/0002.


if __name__ == "__main__":
    main()
