"""Shared test utilities for tensor tests."""

import math

import numpy as np
import numpy.testing as npt

from titan_ai.tensor import Device, Dtype, Tensor
from titan_ai.tensor.backend.numpy_backend import NumpyBackend


def tensor_numpy_array(tensor: Tensor) -> np.ndarray:
    """Return the underlying NumPy array for test verification (internal access)."""
    backend = tensor._backend
    assert isinstance(backend, NumpyBackend)
    return backend.numpy_array


def assert_tensor_equal(tensor: Tensor, expected: np.ndarray) -> None:
    """Assert that a tensor's storage matches a NumPy array."""
    npt.assert_array_equal(tensor_numpy_array(tensor), expected)


def assert_tensor_allclose(tensor: Tensor, expected: np.ndarray, *, rtol: float = 1e-6) -> None:
    """Assert numerical closeness against a NumPy reference array."""
    if expected.dtype == np.float32:
        atol = 1e-6
        rtol = 1e-5
    elif np.issubdtype(expected.dtype, np.floating):
        atol = 1e-12
    else:
        atol = 0
        rtol = 0
    npt.assert_allclose(tensor_numpy_array(tensor), expected, rtol=rtol, atol=atol)


def assert_result_metadata(
    tensor: Tensor,
    expected: np.ndarray,
    dtype: Dtype,
    device: Device | None = None,
) -> None:
    """Assert shape, dtype, device, ndim, and size against a backend result."""
    resolved_device = device if device is not None else Device.cpu()
    expected_shape = tuple(int(dim) for dim in expected.shape)
    assert tensor.shape == expected_shape
    assert tensor.dtype == dtype
    assert tensor.device == resolved_device
    assert tensor.ndim == expected.ndim
    expected_size = 1 if expected_shape == () else int(math.prod(expected_shape))
    assert tensor.size == expected_size
    assert tensor.size == expected.size
