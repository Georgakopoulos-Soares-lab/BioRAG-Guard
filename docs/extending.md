# Adding a corpus

A new corpus is one class. Everything a layer needs goes through
`CorpusAdapter`, so a correct adapter makes every shared layer available
immediately and switches on the modality-specific ones through
`capabilities()`.

## The smallest useful adapter

Suppose a corpus is a directory with `embeddings.npy` and a
`catalog.parquet` carrying an `id` and a `cell_type`:

```python
from pathlib import Path
import numpy as np
import pandas as pd
from biorag_guard.adapters.base import CorpusAdapter, normalise


class MyAtlasAdapter(CorpusAdapter):
    name = "myatlas"
    modality = "cell"

    @classmethod
    def sniff(cls, path: Path) -> float:
        return 0.9 if (path / "embeddings.npy").exists() else 0.0

    def open(self, path, **kw):
        self.path = Path(path)
        self._v = normalise(np.load(self.path / "embeddings.npy", mmap_mode="r"))
        self._df = pd.read_parquet(self.path / "catalog.parquet")
        return self

    @property
    def ids(self):
        return np.asarray(self._df["id"], dtype=object)

    @property
    def vectors(self):
        return self._v

    @property
    def labels(self):
        return np.asarray(self._df["cell_type"], dtype=object)
```

That is enough for `label_consistency`, `dedup`, `hub`,
`duplicate_cluster`, `qc_align`, both manifest layers and the defended
vote. Register it so `bioguard init` finds it:

```toml
[project.entry-points."biorag_guard.adapters"]
myatlas = "mypackage.adapter:MyAtlasAdapter"
```

## What the optional hooks buy

| implement | unlocks |
|---|---|
| `donors` | `donor_cap` at vote time |
| `recompute(records)` | the `recompute` layer &mdash; the only defence against synthesised geometry |
| `neighbors(vectors, k)` | your own ANN index instead of exact cosine |
| `probe_queries(n, seed)` | a probe distribution that matches real usage, which is what `hub` measures against |
| `integrity()` | structural checks only you can make: row counts, key uniqueness, orphaned sidecars |
| `texts` | the text layers |
| `map_integrity()` | `crossmodal_maps` |

`recompute` returning `None` is the correct answer when the corpus cannot
re-derive embeddings. The layer then reports itself as not applicable
rather than passing every record, which is the behaviour the rest of the
package depends on.

## Conventions that are not negotiable

* **Similarity is cosine, descending.** An adapter whose index stores L2
  or `1 - cos` converts in `neighbors`; it does not leak the convention
  upward. (The SCimilarity adapter does exactly this for hnswlib.)
* **Vectors are unit-norm `float32`.** Use `normalise`.
* **`ids` are stable and unique.** If the corpus cannot promise that, say
  so in `integrity()` &mdash; a repeated key means one record silently
  overwrites another, and both shipped adapters check for it.
* **Large corpora are memory-mapped.** Never load a full atlas into RAM;
  `np.load(..., mmap_mode="r")` is enough.

## Testing a new adapter

Write the fixture as a directory in the real layout rather than a mock.
`tests/fixtures.py` builds small SCimilarity- and ProDVa-shaped
directories that way, which is what lets the integrity checks be tested
against the genuine structure. Then assert:

1. `sniff` ranks your adapter first on that directory and zero elsewhere;
2. a planted record is caught, and a clean one is not;
3. whatever your corpus cannot do is *reported*, not silently skipped.
