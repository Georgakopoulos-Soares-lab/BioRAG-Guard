"""Adapter discovery.

Adapters are found through the ``biorag_guard.adapters`` entry-point group,
so a third party adds one without forking. The built-ins are registered
directly as well, so the package works from a source checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.errors import AdapterNotFound
from .base import CorpusAdapter, exact_neighbors, normalise
from .generic import GenericAdapter

_BUILTIN: dict[str, str] = {
    "generic": "biorag_guard.adapters.generic:GenericAdapter",
    "scimilarity": "biorag_guard.adapters.scimilarity:SCimilarityAdapter",
    "prodva": "biorag_guard.adapters.prodva:ProDVaAdapter",
}


def _load(spec: str) -> type[CorpusAdapter]:
    module, _, attr = spec.partition(":")
    mod = __import__(module, fromlist=[attr])
    return getattr(mod, attr)  # type: ignore[no-any-return]


def available() -> dict[str, type[CorpusAdapter]]:
    """Every adapter that can be imported right now, name -> class.

    An adapter whose optional dependency is missing is still listed: the
    class imports, and the missing package is reported when it opens a
    corpus, naming the exact install command.
    """
    out: dict[str, type[CorpusAdapter]] = {}
    for name, spec in _BUILTIN.items():
        try:
            out[name] = _load(spec)
        except Exception:  # pragma: no cover - a broken built-in is a bug
            continue
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group="biorag_guard.adapters"):
            if ep.name not in out:
                try:
                    out[ep.name] = ep.load()
                except Exception:
                    continue
    except Exception:  # pragma: no cover
        pass
    return dict(sorted(out.items()))


def get_adapter(name: str) -> type[CorpusAdapter]:
    adapters = available()
    try:
        return adapters[name]
    except KeyError:
        raise AdapterNotFound(
            f"unknown adapter {name!r}; available: {sorted(adapters)}"
        ) from None


def sniff_all(path: str | Path) -> list[tuple[str, float]]:
    """Every adapter's confidence in this path, best first."""
    path = Path(path)
    scores = []
    for name, cls in available().items():
        try:
            scores.append((name, float(cls.sniff(path))))
        except Exception:
            scores.append((name, 0.0))
    return sorted(scores, key=lambda kv: -kv[1])


def open_corpus(path: str | Path, adapter: str | None = None, **kwargs: Any) -> CorpusAdapter:
    """Open a corpus, detecting the adapter when one is not named."""
    path = Path(path)
    if adapter is not None:
        return get_adapter(adapter)().open(path, **kwargs)
    ranked = sniff_all(path)
    if not ranked or ranked[0][1] <= 0.0:
        tried = ", ".join(
            f"{n} (looked for {_expects(n)})" for n, _ in ranked
        ) or "no adapters are installed"
        raise AdapterNotFound(
            f"nothing at {path} looks like a corpus BioRAG-Guard can read.\n"
            f"  tried: {tried}\n"
            f"  name one explicitly with --adapter, or see docs/extending.md"
        )
    return get_adapter(ranked[0][0])().open(path, **kwargs)


def _expects(name: str) -> str:
    return {
        "generic": "vectors.npy + records.jsonl",
        "scimilarity": "cellsearch/full_kNN_embedding.npy + full_kNN_meta.csv",
        "prodva": "corpus.jsonl / description-sequence + fragment maps",
    }.get(name, "its own layout")


__all__ = [
    "CorpusAdapter", "GenericAdapter", "available", "exact_neighbors", "get_adapter",
    "normalise", "open_corpus", "sniff_all",
]
