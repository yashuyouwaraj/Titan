"""Tests for Day 4 matrix multiplication."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import (
    assert_result_metadata,
    assert_tensor_allclose,
    assert_tensor_equal,
    tensor_numpy_array,
)
from titan_ai.tensor import Device, Dtype, Tensor
from titan_ai.tensor.exceptions.errors import InvalidShapeError, UnsupportedOperandError


class TestMatmul2D:
    def test_valid_inner_dimensions(self) -> None:
        left = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        right = Tensor([[7, 8], [9, 10], [11, 12]], dtype=Dtype.int32)
        result = left @ right
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32) @ np.array(
            [[7, 8], [9, 10], [11, 12]], dtype=np.int32
        )
        assert isinstance(result, Tensor)
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)
        assert result.shape == (2, 2)

    def test_incompatible_inner_dimensions(self) -> None:
        left = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        right = Tensor([[1, 2], [3, 4]], dtype=Dtype.int32)
        with pytest.raises(InvalidShapeError, match=r"\(2, 3\).*\(2, 2\)"):
            _ = left @ right

    def test_float32_numpy_parity(self) -> None:
        left_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        right_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result = Tensor(left_np) @ Tensor(right_np)
        expected = left_np @ right_np
        assert_tensor_allclose(result, expected)
        assert_result_metadata(result, expected, Dtype.float32)


class TestMatmul1D:
    def test_inner_product_returns_scalar_tensor(self) -> None:
        left = Tensor([1, 2, 3], dtype=Dtype.int32)
        right = Tensor([4, 5, 6], dtype=Dtype.int32)
        result = left @ right
        expected = np.array([1, 2, 3], dtype=np.int32) @ np.array([4, 5, 6], dtype=np.int32)
        assert isinstance(result, Tensor)
        assert result.shape == ()
        assert result.ndim == 0
        assert result.size == 1
        assert result.dtype == Dtype.int32
        assert result.device == Device.cpu()
        assert tensor_numpy_array(result) == expected

    def test_incompatible_vector_lengths(self) -> None:
        with pytest.raises(InvalidShapeError, match="inner dimensions"):
            _ = Tensor([1, 2, 3]) @ Tensor([1, 2])

    def test_vector_matrix(self) -> None:
        vector = Tensor([1, 2, 3], dtype=Dtype.int32)
        matrix = Tensor([[1, 2], [3, 4], [5, 6]], dtype=Dtype.int32)
        result = vector @ matrix
        expected = np.array([1, 2, 3], dtype=np.int32) @ np.array(
            [[1, 2], [3, 4], [5, 6]], dtype=np.int32
        )
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)
        assert result.shape == (2,)

    def test_matrix_vector(self) -> None:
        matrix = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        vector = Tensor([7, 8, 9], dtype=Dtype.int32)
        result = matrix @ vector
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32) @ np.array(
            [7, 8, 9], dtype=np.int32
        )
        assert_tensor_equal(result, expected)
        assert result.shape == (2,)


class TestMatmulValidation:
    def test_star_is_not_matmul(self) -> None:
        left = Tensor([[1, 2], [3, 4]], dtype=Dtype.int32)
        right = Tensor([[5, 6], [7, 8]], dtype=Dtype.int32)
        elementwise = left * right
        product = left @ right
        assert not np.array_equal(tensor_numpy_array(elementwise), tensor_numpy_array(product))

    def test_higher_rank_rejected(self) -> None:
        left = Tensor(np.arange(8).reshape(2, 2, 2), dtype=Dtype.int32)
        right = Tensor(np.arange(8).reshape(2, 2, 2), dtype=Dtype.int32)
        with pytest.raises(InvalidShapeError, match="1D and 2D"):
            _ = left @ right

    def test_scalar_tensor_rejected(self) -> None:
        with pytest.raises(InvalidShapeError, match="at least 1D"):
            _ = Tensor(3) @ Tensor([1, 2, 3])

    def test_non_tensor_operand_rejected(self) -> None:
        with pytest.raises(UnsupportedOperandError, match="Tensor operands"):
            _ = Tensor([[1, 2], [3, 4]]) @ [[1, 2], [3, 4]]

    def test_does_not_mutate_operands(self) -> None:
        left = Tensor([[1, 2], [3, 4]], dtype=Dtype.int32)
        right = Tensor([[5, 6], [7, 8]], dtype=Dtype.int32)
        left_copy = tensor_numpy_array(left).copy()
        right_copy = tensor_numpy_array(right).copy()
        _ = left @ right
        assert_tensor_equal(left, left_copy)
        assert_tensor_equal(right, right_copy)

    def test_bool_matmul_promotes_to_int64(self) -> None:
        left = Tensor([[True, False], [False, True]], dtype=Dtype.bool)
        right = Tensor([[True, True], [False, True]], dtype=Dtype.bool)
        result = left @ right
        assert result.dtype == Dtype.int64
        expected = np.array([[True, False], [False, True]], dtype=np.int64) @ np.array(
            [[True, True], [False, True]], dtype=np.int64
        )
        assert_tensor_equal(result, expected)
