"""Screen a submission before it enters a SCimilarity atlas.

The point of the example is how little has to change in an existing
ingest loop: open a guard, ask it, and keep your own indexing code.
"""

from pathlib import Path

import numpy as np

from biorag_guard import Guard, Record, Verdict

ATLAS = Path("/data/scimilarity_model_v1.1")


def main() -> None:
    guard = Guard.auto(ATLAS)
    if guard.baseline is None:
        raise SystemExit(f"run `bioguard calibrate {ATLAS}` first")

    submission = [
        Record(
            id="GSE999_AAACCTGAGAAACCAT",
            vector=np.load("submission/embeddings.npy")[0],
            label="alveolar macrophage",
            donor="donor_17",
            source="GSE999",
        ),
    ]

    report = guard.check_many(submission)
    for result in report.results:
        print(result.record_id, result.verdict.value)
        for finding in result.findings:
            print("   ", finding.reason)
            print("    ->", finding.remedy)

    admitted = [r for r in report.results if r.verdict is Verdict.ADMIT]
    print(f"{len(admitted)}/{len(submission)} admitted")

    # Ingest screening cannot see hub records or edits to existing ones.
    # Schedule this periodically:
    #     bioguard scan /data/scimilarity_model_v1.1


if __name__ == "__main__":
    main()
