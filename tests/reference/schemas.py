"""Minimal shim so the vendored ``label_consistency.py`` imports unchanged.

The upstream module does ``from schemas import EMB_DTYPE``; the real
``schemas.py`` is 42 KB of result-table contracts that the function under
test does not touch. ``EMB_DTYPE`` is ``np.float32`` upstream.
"""

import numpy as np

EMB_DTYPE = np.float32
