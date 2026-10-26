"""One name for "an array", so annotations do not depend on the numpy build.

``np.ndarray`` is a generic type. Whether writing it bare is accepted under
``mypy --strict`` depends on which numpy supplies default type parameters,
so a bare annotation type-checks on one supported Python and fails on
another. Spelling the parameters out once, here, makes the annotations mean
the same thing on every numpy the package supports.

The dtype is deliberately left open. An adapter may hand back float32
vectors, unicode labels or int64 indices, and a layer scores whatever it is
given; pinning a dtype in the annotation would claim a guarantee the
adapter boundary does not make. Where a dtype *is* guaranteed -- unit-norm
float32 out of :func:`~biorag_guard.adapters.base.normalise`, int64 indices
out of :func:`~biorag_guard.adapters.base.exact_neighbors` -- the docstring
says so.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy.typing as npt

#: Any numpy array, of any shape and dtype.
Array: TypeAlias = npt.NDArray[Any]

__all__ = ["Array"]
