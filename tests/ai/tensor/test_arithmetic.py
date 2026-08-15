"""Tests for Day 4 element-wise arithmetic operators."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import (
    assert_result_metadata,
    assert_tensor_allclose,
    assert_tensor_equal,
    tensor_numpy_array,
)
from titan_ai.tensor import Device, Dtype, Tensor
from titan_ai.tensor.exceptions.errors import (
    DeviceMismatchError,
    UnsupportedOperandError,
)


def _numpy_op(op: str, left: np.ndarray, right: np.ndarray | int | float | bool) -> np.ndarray:
    operations = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
    }
    return np.asarray(operations[op](left, right))


class TestAddition:
    def test_tensor_tensor_same_shape(self) -> None:
        left = Tensor([[1, 2], [3, 4]], dtype=Dtype.int32)
        right = Tensor([[5, 6], [7, 8]], dtype=Dtype.int32)
        result = left + right
        expected = np.array([[1, 2], [3, 4]], dtype=np.int32) + np.array(
            [[5, 6], [7, 8]], dtype=np.int32
        )
        assert isinstance(result, Tensor)
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)

    def test_tensor_plus_python_int(self) -> None:
        tensor = Tensor([1, 2, 3], dtype=Dtype.int32)
        result = tensor + 5
        expected = np.array([1, 2, 3], dtype=np.int32) + 5
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)

    def test_python_int_plus_tensor(self) -> None:
        tensor = Tensor([1, 2, 3], dtype=Dtype.int32)
        result = 5 + tensor
        expected = 5 + np.array([1, 2, 3], dtype=np.int32)
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)

    def test_float32_plus_python_float_keeps_float32(self) -> None:
        tensor = Tensor([1.0, 2.0], dtype=Dtype.float32)
        result = tensor + 1.5
        expected = np.array([1.0, 2.0], dtype=np.float32) + np.float32(1.5)
        assert result.dtype == Dtype.float32
        assert_tensor_allclose(result, expected)
        assert_result_metadata(result, expected, Dtype.float32)


class TestSubtraction:
    def test_tensor_tensor(self) -> None:
        left = Tensor([10, 20, 30], dtype=Dtype.int64)
        right = Tensor([1, 2, 3], dtype=Dtype.int64)
        result = left - right
        expected = np.array([10, 20, 30]) - np.array([1, 2, 3])
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int64)

    def test_tensor_minus_scalar(self) -> None:
        tensor = Tensor([10, 20], dtype=Dtype.float64)
        result = tensor - 3
        expected = np.array([10.0, 20.0]) - 3
        assert_tensor_allclose(result, expected)
        assert_result_metadata(result, expected, Dtype.float64)

    def test_scalar_minus_tensor_is_reversed(self) -> None:
        tensor = Tensor([1, 2, 3], dtype=Dtype.int32)
        result = 10 - tensor
        expected = 10 - np.array([1, 2, 3], dtype=np.int32)
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)
        assert not np.array_equal(tensor_numpy_array(result), np.array([1, 2, 3]) - 10)


class TestMultiplication:
    def test_elementwise_not_matmul(self) -> None:
        left = Tensor([[1, 2], [3, 4]], dtype=Dtype.int32)
        right = Tensor([[2, 0], [1, 3]], dtype=Dtype.int32)
        result = left * right
        expected = np.array([[1, 2], [3, 4]], dtype=np.int32) * np.array(
            [[2, 0], [1, 3]], dtype=np.int32
        )
        assert result.shape == (2, 2)
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)

    def test_tensor_times_scalar(self) -> None:
        tensor = Tensor([1.0, 2.0, 3.0], dtype=Dtype.float32)
        result = tensor * 2
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32) * 2
        assert_tensor_allclose(result, expected)
        assert_result_metadata(result, expected, Dtype.float32)

    def test_scalar_times_tensor(self) -> None:
        tensor = Tensor([1, 2, 3], dtype=Dtype.int64)
        result = 4 * tensor
        expected = 4 * np.array([1, 2, 3])
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int64)


class TestDivision:
    def test_tensor_true_div_tensor_integers_to_float64(self) -> None:
        left = Tensor([1, 2, 5], dtype=Dtype.int32)
        right = Tensor([2, 4, 2], dtype=Dtype.int32)
        result = left / right
        expected = np.array([1, 2, 5], dtype=np.int32) / np.array([2, 4, 2], dtype=np.int32)
        assert result.dtype == Dtype.float64
        assert_tensor_allclose(result, expected)
        assert_result_metadata(result, expected, Dtype.float64)

    def test_float32_division_keeps_float32(self) -> None:
        left = Tensor([1.0, 4.0], dtype=Dtype.float32)
        right = Tensor([2.0, 8.0], dtype=Dtype.float32)
        result = left / right
        expected = np.array([1.0, 4.0], dtype=np.float32) / np.array([2.0, 8.0], dtype=np.float32)
        assert result.dtype == Dtype.float32
        assert_tensor_allclose(result, expected)
        assert_result_metadata(result, expected, Dtype.float32)

    def test_tensor_div_scalar(self) -> None:
        tensor = Tensor([2.0, 4.0, 8.0], dtype=Dtype.float64)
        result = tensor / 2
        expected = np.array([2.0, 4.0, 8.0]) / 2
        assert_tensor_allclose(result, expected)
        assert_result_metadata(result, expected, Dtype.float64)

    def test_scalar_div_tensor(self) -> None:
        tensor = Tensor([2.0, 4.0], dtype=Dtype.float64)
        result = 8 / tensor
        expected = 8 / np.array([2.0, 4.0])
        assert_tensor_allclose(result, expected)
        assert_result_metadata(result, expected, Dtype.float64)

    def test_float_division_by_zero_follows_numpy(self) -> None:
        tensor = Tensor([1.0, 0.0, -1.0], dtype=Dtype.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = tensor / 0.0
            expected = np.array([1.0, 0.0, -1.0]) / 0.0
        assert_tensor_allclose(result, expected)
        assert np.isposinf(tensor_numpy_array(result)[0])
        assert np.isnan(tensor_numpy_array(result)[1])
        assert np.isneginf(tensor_numpy_array(result)[2])

    def test_integer_true_div_by_zero_is_inf(self) -> None:
        tensor = Tensor([1, 2], dtype=Dtype.int32)
        with np.errstate(divide="ignore"):
            result = tensor / 0
            expected = np.array([1, 2], dtype=np.int32) / 0
        assert result.dtype == Dtype.float64
        assert_tensor_allclose(result, expected)


class TestNegation:
    def test_negation_creates_new_tensor(self) -> None:
        tensor = Tensor([1, -2, 3], dtype=Dtype.int32)
        result = -tensor
        expected = -np.array([1, -2, 3], dtype=np.int32)
        assert isinstance(result, Tensor)
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)
        assert tensor_numpy_array(tensor)[0] == 1

    def test_bool_negation_promotes_to_int64(self) -> None:
        tensor = Tensor([True, False], dtype=Dtype.bool)
        result = -tensor
        assert result.dtype == Dtype.int64
        assert_tensor_equal(result, np.array([-1, 0], dtype=np.int64))
        assert result.shape == (2,)
        assert result.device == Device.cpu()


class TestOperandIndependence:
    def test_arithmetic_does_not_mutate_operands(self) -> None:
        left = Tensor([1, 2, 3], dtype=Dtype.int32)
        right = Tensor([4, 5, 6], dtype=Dtype.int32)
        original_left = tensor_numpy_array(left).copy()
        original_right = tensor_numpy_array(right).copy()
        _ = left + right
        _ = left - right
        _ = left * right
        _ = left / right
        _ = -left
        assert_tensor_equal(left, original_left)
        assert_tensor_equal(right, original_right)

    def test_result_does_not_alias_input_storage(self) -> None:
        array = np.array([1, 2, 3], dtype=np.int32)
        tensor = Tensor(array, copy=False)
        result = tensor + 1
        array[0] = 99
        assert tensor_numpy_array(result)[0] == 2
        assert tensor_numpy_array(tensor)[0] == 99


class TestUnsupportedOperands:
    def test_tensor_plus_string_raises(self) -> None:
        with pytest.raises(UnsupportedOperandError, match="str"):
            Tensor([1, 2]) + "nope"

    def test_tensor_plus_list_raises(self) -> None:
        with pytest.raises(UnsupportedOperandError, match="list"):
            Tensor([1, 2]) * [1, 2]

    def test_numpy_float16_scalar_rejected(self) -> None:
        with pytest.raises(UnsupportedOperandError):
            Tensor([1.0, 2.0], dtype=Dtype.float32) + np.float16(1.0)


class TestDeviceCompatibility:
    def test_cpu_tensors_operate_together(self) -> None:
        left = Tensor([1, 2], device=Device.cpu())
        right = Tensor([3, 4], device=Device.cpu())
        result = left + right
        assert result.device == Device.cpu()

    def test_device_mismatch_raises(self) -> None:
        left = Tensor([1, 2], dtype=Dtype.int32)
        right = Tensor([3, 4], dtype=Dtype.int32)

        class ForeignDevice:
            type = "cpu"

            def __repr__(self) -> str:
                return "Device('foreign')"

        right._backend._device = ForeignDevice()  # type: ignore[misc]
        with pytest.raises(DeviceMismatchError, match="same device"):
            _ = left + right


@pytest.mark.parametrize(
    ("op", "left_data", "right_data", "dtype"),
    [
        ("add", [1, 2, 3], [4, 5, 6], Dtype.int32),
        ("sub", [10, 20], [1, 2], Dtype.int64),
        ("mul", [1.5, 2.5], [2.0, 3.0], Dtype.float32),
        ("div", [4.0, 9.0], [2.0, 3.0], Dtype.float64),
    ],
)
def test_numpy_parity_same_shape(
    op: str,
    left_data: list[float],
    right_data: list[float],
    dtype: Dtype,
) -> None:
    left = Tensor(left_data, dtype=dtype)
    right = Tensor(right_data, dtype=dtype)
    numpy_left = np.array(left_data, dtype=dtype.to_numpy())
    numpy_right = np.array(right_data, dtype=dtype.to_numpy())
    expected = _numpy_op(op, numpy_left, numpy_right)
    titan_ops = {
        "add": left + right,
        "sub": left - right,
        "mul": left * right,
        "div": left / right,
    }
    result = titan_ops[op]
    if dtype in (Dtype.float32, Dtype.float64) or op == "div":
        assert_tensor_allclose(result, expected)
    else:
        assert_tensor_equal(result, expected)
    assert result.device == Device.cpu()
    assert result.ndim == expected.ndim
    assert result.size == expected.size
    assert result.shape == tuple(expected.shape)
