"""Tests for Day 5 mathematical functions."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import (
    assert_result_metadata,
    assert_tensor_allclose,
    assert_tensor_equal,
    tensor_numpy_array,
)
from titan_ai.tensor import Device, Dtype, Tensor


class TestAbs:
    def test_positive_negative_zero(self) -> None:
        tensor = Tensor([-2, 0, 3], dtype=Dtype.int32)
        result = tensor.abs()
        expected = np.array([2, 0, 3], dtype=np.int32)
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)

    def test_builtin_abs(self) -> None:
        tensor = Tensor([-1.5, 2.0], dtype=Dtype.float64)
        result = abs(tensor)
        assert isinstance(result, Tensor)
        assert_tensor_allclose(result, np.abs(np.array([-1.5, 2.0])))

    def test_float32(self) -> None:
        tensor = Tensor([-1.0, 2.5], dtype=Dtype.float32)
        result = tensor.abs()
        assert result.dtype == Dtype.float32
        assert_tensor_allclose(result, np.abs(np.array([-1.0, 2.5], dtype=np.float32)))

    def test_bool_abs_preserves_bool(self) -> None:
        tensor = Tensor([True, False], dtype=Dtype.bool)
        result = tensor.abs()
        assert result.dtype == Dtype.bool
        assert_tensor_equal(result, np.array([True, False]))

    def test_does_not_mutate(self) -> None:
        tensor = Tensor([-1, 2], dtype=Dtype.int32)
        original = tensor_numpy_array(tensor).copy()
        result = tensor.abs()
        assert_tensor_equal(tensor, original)
        tensor_numpy_array(tensor)[0] = -9
        assert tensor_numpy_array(result)[0] == 1


class TestSqrt:
    def test_zero_and_positive(self) -> None:
        tensor = Tensor([0.0, 4.0, 9.0], dtype=Dtype.float64)
        result = tensor.sqrt()
        expected = np.sqrt(np.array([0.0, 4.0, 9.0]))
        assert_tensor_allclose(result, expected)
        assert_result_metadata(result, expected, Dtype.float64)

    def test_integer_input_returns_float64(self) -> None:
        result = Tensor([0, 4, 9], dtype=Dtype.int32).sqrt()
        assert result.dtype == Dtype.float64
        assert_tensor_allclose(result, np.sqrt(np.array([0, 4, 9], dtype=np.float64)))

    def test_float32(self) -> None:
        tensor = Tensor([1.0, 4.0], dtype=Dtype.float32)
        result = tensor.sqrt()
        assert result.dtype == Dtype.float32
        assert_tensor_allclose(result, np.sqrt(np.array([1.0, 4.0], dtype=np.float32)))

    def test_negative_is_nan_with_warning(self) -> None:
        tensor = Tensor([-1.0, 4.0], dtype=Dtype.float64)
        with pytest.warns(RuntimeWarning):
            result = tensor.sqrt()
        values = tensor_numpy_array(result)
        assert np.isnan(values[0])
        assert values[1] == pytest.approx(2.0)


class TestExp:
    def test_zero_positive_negative(self) -> None:
        data = np.array([0.0, 1.0, -1.0], dtype=np.float64)
        result = Tensor(data).exp()
        expected = np.exp(data)
        assert_tensor_allclose(result, expected)
        assert result.dtype == Dtype.float64
        assert result.device == Device.cpu()

    def test_float32(self) -> None:
        data = np.array([0.0, 0.5], dtype=np.float32)
        result = Tensor(data).exp()
        assert result.dtype == Dtype.float32
        assert_tensor_allclose(result, np.exp(data))

    def test_integer_input_returns_float64(self) -> None:
        result = Tensor([0, 1], dtype=Dtype.int32).exp()
        assert result.dtype == Dtype.float64
        assert_tensor_allclose(result, np.exp(np.array([0.0, 1.0])))

    def test_overflow_follows_numpy(self) -> None:
        tensor = Tensor([0.0, 1000.0], dtype=Dtype.float64)
        data = np.array([0.0, 1000.0])
        with pytest.warns(RuntimeWarning):
            result = tensor.exp()
            expected = np.exp(data)
        assert np.isinf(tensor_numpy_array(result)[1])
        assert_tensor_allclose(result, expected)


class TestLog:
    def test_positive_and_one(self) -> None:
        data = np.array([1.0, np.e, 4.0], dtype=np.float64)
        result = Tensor(data).log()
        assert_tensor_allclose(result, np.log(data))
        assert result.dtype == Dtype.float64

    def test_fractional(self) -> None:
        data = np.array([0.5], dtype=np.float64)
        result = Tensor(data).log()
        assert_tensor_allclose(result, np.log(data))

    def test_float32(self) -> None:
        data = np.array([1.0, 2.0], dtype=np.float32)
        result = Tensor(data).log()
        assert result.dtype == Dtype.float32
        assert_tensor_allclose(result, np.log(data))

    def test_zero_is_neginf_with_warning(self) -> None:
        tensor = Tensor([0.0, 1.0], dtype=Dtype.float64)
        with pytest.warns(RuntimeWarning):
            result = tensor.log()
        values = tensor_numpy_array(result)
        assert np.isneginf(values[0])
        assert values[1] == pytest.approx(0.0)

    def test_negative_is_nan_with_warning(self) -> None:
        tensor = Tensor([-1.0, 1.0], dtype=Dtype.float64)
        with pytest.warns(RuntimeWarning):
            result = tensor.log()
        values = tensor_numpy_array(result)
        assert np.isnan(values[0])
        assert values[1] == pytest.approx(0.0)

    def test_integer_input_returns_float64(self) -> None:
        result = Tensor([1, 1], dtype=Dtype.int32).log()
        assert result.dtype == Dtype.float64
        assert_tensor_allclose(result, np.log(np.array([1.0, 1.0])))
