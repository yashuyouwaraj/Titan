"""Shared test utilities for tensor tests."""

import numpy as np
import numpy.testing as npt

from titan_ai.tensor import Tensor
from titan_ai.tensor.backend.numpy_backend import NumpyBackend


def tensor_numpy_array(tensor: Tensor) -> np.ndarray:
    """Return the underlying NumPy array for test verification (internal access)."""
    backend = tensor._backend
    assert isinstance(backend, NumpyBackend)
    return backend.numpy_array


def assert_tensor_equal(tensor: Tensor, expected: np.ndarray) -> None:
    """Assert that a tensor's storage matches a NumPy array."""
    npt.assert_array_equal(tensor_numpy_array(tensor), expected)
