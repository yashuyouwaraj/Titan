"""Tensor shape operation helpers."""

from titan_ai.tensor.operations.axis import normalize_reduction_axis
from titan_ai.tensor.operations.broadcast import broadcast_shapes
from titan_ai.tensor.operations.shape import normalize_shape, resolve_inferred_shape

__all__ = [
    "broadcast_shapes",
    "normalize_reduction_axis",
    "normalize_shape",
    "resolve_inferred_shape",
]
