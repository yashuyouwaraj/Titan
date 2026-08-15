"""Regression tests for 0-d tensors discovered during the Day 6 audit."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import (
    assert_result_metadata,
    assert_tensor_allclose,
    assert_tensor_equal,
)
from titan_ai.tensor import Dtype, Tensor
from titan_ai.tensor.exceptions.errors import InvalidShapeError


class TestZeroDimArithmetic:
    def test_scalar_tensor_plus_python_int(self) -> None:
        tensor = Tensor(3, dtype=Dtype.int32)
        result = tensor + 1
        expected = np.array(4, dtype=np.int32)
        assert isinstance(result, Tensor)
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)

    def test_scalar_tensor_plus_scalar_tensor(self) -> None:
        result = Tensor(3, dtype=Dtype.int32) + Tensor(4, dtype=Dtype.int32)
        assert_tensor_equal(result, np.array(7, dtype=np.int32))
        assert result.shape == ()

    def test_reverse_and_mul_div(self) -> None:
        tensor = Tensor(8, dtype=Dtype.int32)
        assert_tensor_equal(10 - tensor, np.array(2, dtype=np.int32))
        assert_tensor_equal(tensor * 2, np.array(16, dtype=np.int32))
        result = tensor / 2
        assert result.dtype == Dtype.float64
        assert_tensor_allclose(result, np.array(4.0))

    def test_scalar_broadcasts_to_vector(self) -> None:
        result = Tensor(3, dtype=Dtype.int32) + Tensor([1, 2, 3], dtype=Dtype.int32)
        assert_tensor_equal(result, np.array([4, 5, 6], dtype=np.int32))


class TestZeroDimMath:
    def test_negation_and_abs(self) -> None:
        tensor = Tensor(-4, dtype=Dtype.int32)
        assert_tensor_equal(-tensor, np.array(4, dtype=np.int32))
        assert_tensor_equal(tensor.abs(), np.array(4, dtype=np.int32))
        assert (-tensor).shape == ()

    def test_sqrt_exp_log(self) -> None:
        assert_tensor_allclose(Tensor(9.0).sqrt(), np.array(3.0))
        assert_tensor_allclose(Tensor(0.0).exp(), np.array(1.0))
        assert_tensor_allclose(Tensor(1.0).log(), np.array(0.0))
        assert Tensor(9.0).sqrt().shape == ()


class TestZeroDimFactoriesAndReshape:
    def test_zeros_empty_tuple_is_scalar_tensor(self) -> None:
        tensor = Tensor.zeros((), dtype=Dtype.float32)
        assert tensor.shape == ()
        assert tensor.ndim == 0
        assert tensor.size == 1
        assert_tensor_equal(tensor, np.array(0.0, dtype=np.float32))

    def test_ones_empty_tuple(self) -> None:
        tensor = Tensor.ones(())
        assert tensor.shape == ()
        assert_tensor_equal(tensor, np.array(1.0))

    def test_reshape_size_one_to_scalar(self) -> None:
        tensor = Tensor([7], dtype=Dtype.int32).reshape(())
        assert tensor.shape == ()
        assert_tensor_equal(tensor, np.array(7, dtype=np.int32))

    def test_reshape_non_unit_size_to_scalar_raises(self) -> None:
        with pytest.raises(InvalidShapeError):
            Tensor([1, 2, 3]).reshape(())
