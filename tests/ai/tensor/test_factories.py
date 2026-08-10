"""Tests for Tensor factory methods."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import assert_tensor_equal, tensor_numpy_array
from titan_ai.tensor import Device, Dtype, Tensor
from titan_ai.tensor.exceptions.errors import InvalidShapeError, TensorConstructionError


class TestZeros:
    def test_zeros_2d(self) -> None:
        tensor = Tensor.zeros((2, 3))
        assert tensor.shape == (2, 3)
        assert tensor.dtype == Dtype.float64
        assert tensor.device == Device.cpu()
        assert_tensor_equal(tensor, np.zeros((2, 3)))

    def test_zeros_with_dtype(self) -> None:
        tensor = Tensor.zeros((2, 3), dtype=Dtype.float32)
        assert tensor.dtype == Dtype.float32
        assert_tensor_equal(tensor, np.zeros((2, 3), dtype=np.float32))


class TestOnes:
    def test_ones_2d(self) -> None:
        tensor = Tensor.ones((2, 3))
        assert tensor.shape == (2, 3)
        assert_tensor_equal(tensor, np.ones((2, 3)))


class TestEmpty:
    def test_empty_allocation(self) -> None:
        tensor = Tensor.empty((2, 3), dtype=Dtype.float32)
        assert tensor.shape == (2, 3)
        assert tensor.dtype == Dtype.float32
        assert tensor.device == Device.cpu()
        assert tensor.size == 6


class TestFull:
    def test_full_int(self) -> None:
        tensor = Tensor.full((2, 3), 7)
        assert tensor.shape == (2, 3)
        assert_tensor_equal(tensor, np.full((2, 3), 7))

    def test_full_float_dtype(self) -> None:
        tensor = Tensor.full((2, 3), 3.14, dtype=Dtype.float32)
        assert tensor.dtype == Dtype.float32
        assert_tensor_equal(tensor, np.full((2, 3), 3.14, dtype=np.float32))


class TestArange:
    def test_arange_stop_only(self) -> None:
        tensor = Tensor.arange(5)
        assert tensor.shape == (5,)
        assert_tensor_equal(tensor, np.arange(5))

    def test_arange_start_stop(self) -> None:
        tensor = Tensor.arange(2, 5)
        assert_tensor_equal(tensor, np.arange(2, 5))

    def test_arange_with_step(self) -> None:
        tensor = Tensor.arange(0, 10, 2)
        assert_tensor_equal(tensor, np.arange(0, 10, 2))

    def test_arange_zero_step_raises(self) -> None:
        with pytest.raises(TensorConstructionError, match="step"):
            Tensor.arange(0, 5, 0)


class TestFromNumpy:
    def test_from_numpy_copies_by_default(self) -> None:
        array = np.array([1, 2, 3], dtype=np.int32)
        tensor = Tensor.from_numpy(array)
        array[0] = 99
        assert tensor_numpy_array(tensor)[0] == 1

    def test_from_numpy_zero_copy(self) -> None:
        array = np.array([1, 2, 3], dtype=np.int32)
        tensor = Tensor.from_numpy(array, copy=False)
        array[0] = 99
        assert tensor_numpy_array(tensor)[0] == 99


class TestFactoryValidation:
    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(InvalidShapeError):
            Tensor.zeros((2, -2))
