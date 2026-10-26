"""Per-neighbour vote weights.

Ported from ``00_common/defense.py`` of the Single_Cell_Poisoning study
(MIT). The schemes, their formulas and the score clip are kept exactly;
``tests/test_vote_equivalence.py`` pins this module against the original on
random inputs.

Weighting is the one policy here that is not a filter: nothing is rejected,
the arithmetic of the tally changes. Whether it helps is not obvious in
advance -- a cluster flood places its records CLOSER to the query than
genuine neighbours are, so proximity weighting can amplify the very attack
it is proposed against. Measure it on your corpus; do not assume it.
"""

from __future__ import annotations

import numpy as np

from ..core.arrays import Array

#: Floor on ``1 - cos`` in the score scheme. A record copied verbatim can
#: land at cos = 1.0 in float32, making ``1 / (1 - c)`` infinite, and one
#: such neighbour would silently decide every vote. The clip is an explicit
#: modelling choice, not a silent numerical fallback: 1e-6 caps the weight
#: at 1e6, far above anything a genuine neighbour attains.
SCORE_EPS = 1e-6

WEIGHT_SCHEMES = ("uniform", "score", "cosine", "rank", "softmax")


def vote_weights(cos: Array, scheme: str, temp: float = 0.01) -> Array:
    """Weights for the surviving neighbours of one query.

    Args:
        cos: (m,) cosine similarities in rank order -- descending, so index
            0 is the nearest.
        scheme: one of :data:`WEIGHT_SCHEMES`.
        temp: softmax temperature; ignored by the other schemes.

    Returns:
        (m,) float64 non-negative weights. ``uniform`` returns all ones,
        which reproduces the plain majority vote exactly.

    Raises:
        ValueError: unknown scheme. Falling back to uniform would report a
            weighting that never ran as a weighting that made no difference.
    """
    c = np.asarray(cos, dtype=np.float64)
    if scheme == "uniform":
        return np.ones(len(c))
    if scheme == "score":
        # the SCimilarity score itself, 1 / (1 - c)
        return 1.0 / np.maximum(1.0 - c, SCORE_EPS)
    if scheme == "cosine":
        # shifted so an antipodal neighbour contributes nothing rather than
        # a negative vote, which is not meaningful in a tally
        return np.maximum(c, 0.0)
    if scheme == "rank":
        return 1.0 / (1.0 + np.arange(len(c)))
    if scheme == "softmax":
        z = (c - c.max()) / temp
        return np.asarray(np.exp(z), dtype=np.float64)
    raise ValueError(f"unknown weight scheme {scheme!r}; expected one of {WEIGHT_SCHEMES}")
