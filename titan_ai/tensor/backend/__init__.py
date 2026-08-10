"""Tensor storage backends."""

from titan_ai.tensor.backend.base import TensorBackend
from titan_ai.tensor.backend.numpy_backend import NumpyBackend

__all__ = ["TensorBackend", "NumpyBackend"]
