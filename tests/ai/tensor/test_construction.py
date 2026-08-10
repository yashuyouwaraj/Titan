"""Tests for Tensor construction."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import tensor_numpy_array
from titan_ai.tensor import Device, Dtype, Tensor
from titan_ai.tensor.exceptions.errors import TensorConstructionError


class TestTensorConstruction:
    def test_scalar_int(self) -> None:
        tensor = Tensor(7)
        assert tensor.shape == ()
        assert tensor.ndim == 0
        assert tensor.size == 1

    def test_scalar_float(self) -> None:
        tensor = Tensor(3.14)
        assert tensor.shape == ()
        assert tensor.size == 1

    def test_1d_list(self) -> None:
        tensor = Tensor([1, 2, 3])
        assert tensor.shape == (3,)
        assert tensor.ndim == 1
        assert tensor.size == 3

    def test_2d_nested_list(self) -> None:
        tensor = Tensor([[1, 2], [3, 4]])
        assert tensor.shape == (2, 2)
        assert tensor.ndim == 2
        assert tensor.size == 4

    def test_3d_nested_list(self) -> None:
        tensor = Tensor([[1, 2], [3, 4], [5, 6]])
        assert tensor.shape == (3, 2)
        assert tensor.size == 6

    def test_numpy_array_input(self) -> None:
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = Tensor(array)
        assert tensor.shape == (3,)
        assert tensor.dtype == Dtype.float32

    def test_explicit_dtype_override(self) -> None:
        tensor = Tensor([1, 2, 3], dtype=Dtype.float32)
        assert tensor.dtype == Dtype.float32

    def test_explicit_device_cpu(self) -> None:
        tensor = Tensor([1, 2], device=Device.cpu())
        assert tensor.device == Device.cpu()

    def test_invalid_nested_structure_raises(self) -> None:
        with pytest.raises(TensorConstructionError):
            Tensor([[1, 2], [3]])

    def test_empty_list(self) -> None:
        tensor = Tensor([])
        assert tensor.shape == (0,)
        assert tensor.size == 0

    def test_values_match_input_lists(self) -> None:
        tensor = Tensor([[1, 2], [3, 4]])
        assert np.array_equal(tensor_numpy_array(tensor), np.array([[1, 2], [3, 4]]))
