"""Tests for Day 4 dtype promotion rules."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import assert_tensor_allclose, assert_tensor_equal
from titan_ai.tensor import Dtype, Tensor
from titan_ai.tensor.operations.promotion import promote_dtypes, promote_tensor_and_scalar


class TestTensorTensorPromotion:
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (Dtype.int32, Dtype.int32, Dtype.int32),
            (Dtype.int32, Dtype.int64, Dtype.int64),
            (Dtype.int32, Dtype.float32, Dtype.float64),
            (Dtype.int32, Dtype.float64, Dtype.float64),
            (Dtype.int64, Dtype.float32, Dtype.float64),
            (Dtype.float32, Dtype.float32, Dtype.float32),
            (Dtype.float32, Dtype.float64, Dtype.float64),
            (Dtype.bool, Dtype.int32, Dtype.int32),
            (Dtype.bool, Dtype.float32, Dtype.float32),
        ],
    )
    def test_promotion_table(self, left: Dtype, right: Dtype, expected: Dtype) -> None:
        assert promote_dtypes(left, right) == expected
        assert promote_dtypes(right, left) == expected

    def test_int_plus_int_keeps_int32(self) -> None:
        result = Tensor([1, 2], dtype=Dtype.int32) + Tensor([3, 4], dtype=Dtype.int32)
        assert result.dtype == Dtype.int32
        assert_tensor_equal(result, np.array([4, 6], dtype=np.int32))

    def test_int_plus_float_promotes_to_float64(self) -> None:
        result = Tensor([1, 2], dtype=Dtype.int32) + Tensor([0.5, 1.5], dtype=Dtype.float32)
        assert result.dtype == Dtype.float64
        expected = np.array([1, 2], dtype=np.int32) + np.array([0.5, 1.5], dtype=np.float32)
        assert_tensor_allclose(result, expected)

    def test_float32_plus_float32(self) -> None:
        result = Tensor([1.0], dtype=Dtype.float32) + Tensor([2.0], dtype=Dtype.float32)
        assert result.dtype == Dtype.float32

    def test_float32_plus_float64(self) -> None:
        result = Tensor([1.0], dtype=Dtype.float32) + Tensor([2.0], dtype=Dtype.float64)
        assert result.dtype == Dtype.float64


class TestScalarPromotion:
    def test_python_int_does_not_widen_int32(self) -> None:
        result = Tensor([1, 2], dtype=Dtype.int32) + 1
        assert result.dtype == Dtype.int32
        assert promote_tensor_and_scalar(Dtype.int32, 1) == Dtype.int32

    def test_python_float_does_not_widen_float32(self) -> None:
        result = Tensor([1.0], dtype=Dtype.float32) + 1.0
        assert result.dtype == Dtype.float32

    def test_python_float_promotes_int_to_float64(self) -> None:
        result = Tensor([1, 2], dtype=Dtype.int32) + 1.0
        assert result.dtype == Dtype.float64

    def test_python_int_promotes_bool_to_int64(self) -> None:
        result = Tensor([True, False], dtype=Dtype.bool) + 1
        assert result.dtype == Dtype.int64
        assert_tensor_equal(result, np.array([2, 1], dtype=np.int64))

    def test_numpy_float64_scalar_is_strong(self) -> None:
        result = Tensor([1.0], dtype=Dtype.float32) + np.float64(1.0)
        assert result.dtype == Dtype.float64

    def test_numpy_int32_scalar_is_strong(self) -> None:
        result = Tensor([1], dtype=Dtype.int64) + np.int32(1)
        assert result.dtype == Dtype.int64
        assert promote_tensor_and_scalar(Dtype.int64, np.int32(1)) == Dtype.int64
