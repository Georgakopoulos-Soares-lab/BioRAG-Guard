"""Every layer, registered on import.

Importing this package is what populates the registry, so the CLI, the
docs generator and ``Guard`` all see the same set.
"""

from ..core.registry import Layer, LayerSpec, all_layers, get_layer, layers_for, register_layer
from . import cell, crossmodal, integrity, population, qc, text, vote_time  # noqa: F401
from .scoring import max_similarity, score_records

__all__ = [
    "Layer", "LayerSpec", "all_layers", "get_layer", "layers_for", "max_similarity",
    "register_layer", "score_records",
]
