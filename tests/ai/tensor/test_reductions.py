"""Tests for Day 5 reductions."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import (
    assert_result_metadata,
    assert_tensor_allclose,
    assert_tensor_equal,
    tensor_numpy_array,
)
from titan_ai.tensor import Device, Dtype, Tensor
from titan_ai.tensor.exceptions.errors import InvalidAxisError, TensorValidationError


class TestSum:
    def test_vector(self) -> None:
        tensor = Tensor([1, 2, 3], dtype=Dtype.int32)
        result = tensor.sum()
        expected = np.array(6, dtype=np.int64)
        assert isinstance(result, Tensor)
        assert result.shape == ()
        assert result.ndim == 0
        assert result.size == 1
        assert result.dtype == Dtype.int64
        assert result.device == Device.cpu()
        assert_tensor_equal(result, expected)

    def test_scalar(self) -> None:
        result = Tensor(7, dtype=Dtype.int32).sum()
        assert result.shape == ()
        assert_tensor_equal(result, np.array(7, dtype=np.int64))

    def test_matrix_axis_none(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        result = tensor.sum()
        assert_tensor_equal(result, np.array(21, dtype=np.int64))

    def test_matrix_axis_0(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        result = tensor.sum(axis=0)
        expected = np.array([5, 7, 9], dtype=np.int64)
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int64)

    def test_matrix_axis_1(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        result = tensor.sum(axis=1)
        expected = np.array([6, 15], dtype=np.int64)
        assert_tensor_equal(result, expected)
        assert result.shape == (2,)

    def test_negative_axis(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        result = tensor.sum(axis=-1)
        expected = tensor.sum(axis=1)
        assert_tensor_equal(result, tensor_numpy_array(expected))

    def test_keepdims(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        result = tensor.sum(axis=1, keepdims=True)
        expected = np.array([[6], [15]], dtype=np.int64)
        assert_tensor_equal(result, expected)
        assert result.shape == (2, 1)

    def test_keepdims_axis_none(self) -> None:
        tensor = Tensor([[1, 2], [3, 4]], dtype=Dtype.int32)
        result = tensor.sum(keepdims=True)
        assert result.shape == (1, 1)
        assert_tensor_equal(result, np.array([[10]], dtype=np.int64))

    def test_3d(self) -> None:
        array = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
        tensor = Tensor(array)
        result = tensor.sum(axis=0)
        expected = array.sum(axis=0, dtype=np.int64)
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int64)

    def test_float32_keeps_float32(self) -> None:
        tensor = Tensor([1.0, 2.0, 3.0], dtype=Dtype.float32)
        result = tensor.sum()
        assert result.dtype == Dtype.float32
        assert_tensor_allclose(result, np.array(6.0, dtype=np.float32))

    def test_bool_promotes_to_int64(self) -> None:
        tensor = Tensor([True, False, True], dtype=Dtype.bool)
        result = tensor.sum()
        assert result.dtype == Dtype.int64
        assert_tensor_equal(result, np.array(2, dtype=np.int64))

    def test_does_not_mutate_input(self) -> None:
        tensor = Tensor([1, 2, 3], dtype=Dtype.int32)
        original = tensor_numpy_array(tensor).copy()
        _ = tensor.sum()
        assert_tensor_equal(tensor, original)


class TestMean:
    def test_integer_input_returns_float64(self) -> None:
        tensor = Tensor([1, 2, 3], dtype=Dtype.int32)
        result = tensor.mean()
        assert result.dtype == Dtype.float64
        assert_tensor_allclose(result, np.array(2.0, dtype=np.float64))
        assert result.shape == ()

    def test_float32_keeps_float32(self) -> None:
        tensor = Tensor([1.0, 3.0], dtype=Dtype.float32)
        result = tensor.mean()
        assert result.dtype == Dtype.float32
        assert_tensor_allclose(result, np.array(2.0, dtype=np.float32))

    def test_float64(self) -> None:
        tensor = Tensor([2.0, 4.0, 6.0], dtype=Dtype.float64)
        result = tensor.mean()
        assert result.dtype == Dtype.float64
        assert_tensor_allclose(result, np.array(4.0))

    def test_scalar(self) -> None:
        result = Tensor(8, dtype=Dtype.int64).mean()
        assert result.dtype == Dtype.float64
        assert_tensor_allclose(result, np.array(8.0))

    def test_matrix_axis(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        result = tensor.mean(axis=0)
        expected = np.array([2.5, 3.5, 4.5], dtype=np.float64)
        assert_tensor_allclose(result, expected)
        assert_result_metadata(result, expected, Dtype.float64)

    def test_negative_axis_keepdims(self) -> None:
        tensor = Tensor([[1.0, 3.0], [5.0, 7.0]], dtype=Dtype.float64)
        result = tensor.mean(axis=-1, keepdims=True)
        expected = np.array([[2.0], [6.0]])
        assert_tensor_allclose(result, expected)
        assert result.shape == (2, 1)

    def test_bool_mean_is_float64(self) -> None:
        result = Tensor([True, True, False], dtype=Dtype.bool).mean()
        assert result.dtype == Dtype.float64
        assert_tensor_allclose(result, np.array(2.0 / 3.0))


class TestMinMax:
    @pytest.mark.parametrize("op", ["min", "max"])
    def test_vector(self, op: str) -> None:
        tensor = Tensor([3, -1, 4], dtype=Dtype.int32)
        result = getattr(tensor, op)()
        expected = np.array(-1 if op == "min" else 4, dtype=np.int32)
        assert result.dtype == Dtype.int32
        assert result.shape == ()
        assert_tensor_equal(result, expected)

    def test_matrix_axis(self) -> None:
        tensor = Tensor([[1, 8, 3], [4, 2, 9]], dtype=Dtype.int32)
        assert_tensor_equal(tensor.min(axis=0), np.array([1, 2, 3], dtype=np.int32))
        assert_tensor_equal(tensor.max(axis=0), np.array([4, 8, 9], dtype=np.int32))
        assert_tensor_equal(tensor.min(axis=1), np.array([1, 2], dtype=np.int32))
        assert_tensor_equal(tensor.max(axis=1), np.array([8, 9], dtype=np.int32))

    def test_keepdims_and_negative_axis(self) -> None:
        tensor = Tensor([[1.0, -2.0], [3.0, 0.5]], dtype=Dtype.float32)
        result = tensor.min(axis=-1, keepdims=True)
        expected = np.array([[-2.0], [0.5]], dtype=np.float32)
        assert_tensor_allclose(result, expected)
        assert result.shape == (2, 1)
        assert result.dtype == Dtype.float32

    def test_scalar(self) -> None:
        tensor = Tensor(5, dtype=Dtype.int64)
        assert_tensor_equal(tensor.min(), np.array(5, dtype=np.int64))
        assert_tensor_equal(tensor.max(), np.array(5, dtype=np.int64))

    def test_bool(self) -> None:
        tensor = Tensor([True, False, True], dtype=Dtype.bool)
        assert tensor.min().dtype == Dtype.bool
        assert tensor.max().dtype == Dtype.bool
        assert_tensor_equal(tensor.min(), np.array(False))
        assert_tensor_equal(tensor.max(), np.array(True))

    def test_does_not_alias_storage(self) -> None:
        array = np.array([1, 2, 3], dtype=np.int32)
        tensor = Tensor(array, copy=False)
        result = tensor.min()
        array[0] = 99
        assert tensor_numpy_array(result) == 1


class TestAxisValidation:
    def test_valid_positive_and_negative(self) -> None:
        tensor = Tensor(np.arange(24).reshape(2, 3, 4), dtype=Dtype.int32)
        assert tensor.sum(axis=2).shape == (2, 3)
        assert tensor.sum(axis=-1).shape == (2, 3)
        assert tensor.sum(axis=-3).shape == (3, 4)

    def test_invalid_positive_axis(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(InvalidAxisError, match="out of bounds"):
            tensor.sum(axis=3)
        with pytest.raises(InvalidAxisError, match="out of bounds"):
            tensor.mean(axis=2)

    def test_invalid_negative_axis(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(InvalidAxisError, match="out of bounds"):
            tensor.min(axis=-3)

    def test_non_integer_axis(self) -> None:
        tensor = Tensor([1, 2, 3])
        with pytest.raises(InvalidAxisError, match="integer"):
            tensor.max(axis=1.5)  # type: ignore[arg-type]

    def test_tuple_axes_not_supported(self) -> None:
        tensor = Tensor(np.arange(24).reshape(2, 3, 4), dtype=Dtype.int32)
        with pytest.raises(InvalidAxisError):
            tensor.sum(axis=(0, 1))  # type: ignore[arg-type]

    def test_zero_d_axis_must_be_none(self) -> None:
        with pytest.raises(InvalidAxisError, match="0-dimensional"):
            Tensor(3).sum(axis=0)


class TestEmptyReductions:
    def test_sum_of_empty_is_zero(self) -> None:
        result = Tensor.zeros((0,), dtype=Dtype.int32).sum()
        assert result.dtype == Dtype.int64
        assert_tensor_equal(result, np.array(0, dtype=np.int64))

    def test_sum_along_empty_axis(self) -> None:
        result = Tensor.zeros((2, 0), dtype=Dtype.int32).sum(axis=1)
        assert_tensor_equal(result, np.array([0, 0], dtype=np.int64))

    def test_mean_of_empty_raises(self) -> None:
        with pytest.raises(TensorValidationError, match="empty reduction"):
            Tensor.zeros((0,), dtype=Dtype.float64).mean()

    def test_min_of_empty_raises(self) -> None:
        with pytest.raises(TensorValidationError, match="empty reduction"):
            Tensor.zeros((2, 0), dtype=Dtype.int32).min(axis=1)

    def test_max_of_empty_raises(self) -> None:
        with pytest.raises(TensorValidationError, match="empty reduction"):
            Tensor.zeros((0, 3), dtype=Dtype.int32).max(axis=0)

    def test_min_along_nonempty_axis_of_empty_outer_is_empty(self) -> None:
        result = Tensor.zeros((0, 3), dtype=Dtype.int32).min(axis=1)
        assert result.shape == (0,)
        assert result.dtype == Dtype.int32
