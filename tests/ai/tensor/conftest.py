"""Shared test utilities for tensor tests."""

import numpy as np

from titan_ai.tensor import Tensor
from titan_ai.tensor.backend.numpy_backend import NumpyBackend


def tensor_numpy_array(tensor: Tensor) -> np.ndarray:
    """Return the underlying NumPy array for test verification (internal access)."""
    backend = tensor._backend
    assert isinstance(backend, NumpyBackend)
    return backend.numpy_array
