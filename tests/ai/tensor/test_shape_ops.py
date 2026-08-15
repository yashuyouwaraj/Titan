"""Tests for Tensor shape operations."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import assert_tensor_equal, tensor_numpy_array
from titan_ai.tensor import Dtype, Tensor
from titan_ai.tensor.exceptions.errors import (
    InvalidAxisError,
    InvalidShapeError,
    TensorValidationError,
)


class TestReshape:
    def test_reshape_tuple(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]]).reshape((3, 2))
        assert tensor.shape == (3, 2)
        assert_tensor_equal(tensor, np.array([[1, 2, 3], [4, 5, 6]]).reshape(3, 2))

    def test_reshape_args(self) -> None:
        tensor = Tensor([1, 2, 3, 4, 5, 6]).reshape(2, 3)
        assert tensor.shape == (2, 3)

    def test_reshape_infer_dimension(self) -> None:
        tensor = Tensor([1, 2, 3, 4, 5, 6]).reshape(2, -1)
        assert tensor.shape == (2, 3)

    def test_reshape_invalid_size_raises(self) -> None:
        with pytest.raises(InvalidShapeError):
            Tensor([1, 2, 3]).reshape(2, 2)

    def test_reshape_multiple_infer_raises(self) -> None:
        with pytest.raises(InvalidShapeError):
            Tensor([1, 2, 3, 4]).reshape(-1, -1)

    def test_reshape_preserves_dtype_device(self) -> None:
        original = Tensor([1, 2, 3, 4], dtype=Dtype.float32)
        reshaped = original.reshape(2, 2)
        assert reshaped.dtype == Dtype.float32
        assert reshaped.device == original.device


class TestTranspose:
    def test_transpose_2d(self) -> None:
        base = np.array([[1, 2, 3], [4, 5, 6]])
        tensor = Tensor(base).transpose()
        assert tensor.shape == (3, 2)
        assert_tensor_equal(tensor, base.T)

    def test_transpose_axes(self) -> None:
        tensor = Tensor(np.arange(8).reshape(2, 2, 2)).transpose(2, 0, 1)
        expected = np.arange(8).reshape(2, 2, 2).transpose(2, 0, 1)
        assert_tensor_equal(tensor, expected)

    def test_T_property(self) -> None:
        tensor = Tensor([[1, 2], [3, 4]]).T
        assert tensor.shape == (2, 2)
        assert_tensor_equal(tensor, np.array([[1, 2], [3, 4]]).T)

    def test_T_invalid_ndim_raises(self) -> None:
        with pytest.raises(TensorValidationError):
            _ = Tensor([1, 2, 3]).T


class TestFlatten:
    def test_flatten(self) -> None:
        tensor = Tensor([[1, 2], [3, 4]]).flatten()
        assert tensor.shape == (4,)
        assert_tensor_equal(tensor, np.array([[1, 2], [3, 4]]).flatten())


class TestSqueeze:
    def test_squeeze_all(self) -> None:
        tensor = Tensor(np.ones((1, 3, 1, 4))).squeeze()
        assert tensor.shape == (3, 4)

    def test_squeeze_axis(self) -> None:
        tensor = Tensor(np.ones((1, 3, 4))).squeeze(axis=0)
        assert tensor.shape == (3, 4)

    def test_squeeze_invalid_axis_raises(self) -> None:
        with pytest.raises(InvalidAxisError):
            Tensor(np.ones((3, 4))).squeeze(axis=0)


class TestUnsqueeze:
    def test_unsqueeze_axis_0(self) -> None:
        tensor = Tensor(np.arange(12).reshape(3, 4)).unsqueeze(0)
        assert tensor.shape == (1, 3, 4)

    def test_unsqueeze_negative_axis(self) -> None:
        tensor = Tensor(np.arange(12).reshape(3, 4)).unsqueeze(-1)
        assert tensor.shape == (3, 4, 1)

    def test_unsqueeze_invalid_axis_raises(self) -> None:
        with pytest.raises(InvalidAxisError):
            Tensor([1, 2, 3]).unsqueeze(5)


class TestShapeMemorySemantics:
    def test_reshape_may_share_storage(self) -> None:
        original = Tensor(np.arange(6))
        reshaped = original.reshape(2, 3)
        tensor_numpy_array(original)[0] = 99
        assert tensor_numpy_array(reshaped)[0, 0] == 99

    def test_flatten_creates_copy(self) -> None:
        original = Tensor([[1, 2], [3, 4]])
        flat = original.flatten()
        tensor_numpy_array(original)[0, 0] = 99
        assert tensor_numpy_array(flat)[0] == 1

    def test_transpose_is_view(self) -> None:
        original = Tensor(np.arange(6).reshape(2, 3))
        transposed = original.transpose()
        tensor_numpy_array(original)[0, 0] = 99
        assert tensor_numpy_array(transposed)[0, 0] == 99

    def test_squeeze_is_view(self) -> None:
        original = Tensor(np.arange(3).reshape(1, 3))
        squeezed = original.squeeze()
        tensor_numpy_array(original)[0, 0] = 7
        assert tensor_numpy_array(squeezed)[0] == 7
