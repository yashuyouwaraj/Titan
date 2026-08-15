"""Tests for Day 4 broadcasting validation and arithmetic."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import (
    assert_result_metadata,
    assert_tensor_allclose,
    assert_tensor_equal,
)
from titan_ai.tensor import Device, Dtype, Tensor
from titan_ai.tensor.exceptions.errors import BroadcastError
from titan_ai.tensor.operations.broadcast import broadcast_shapes


class TestBroadcastShapes:
    def test_identical_vectors(self) -> None:
        assert broadcast_shapes((3,), (3,)) == (3,)

    def test_matrix_and_vector(self) -> None:
        assert broadcast_shapes((2, 3), (3,)) == (2, 3)

    def test_singleton_row(self) -> None:
        assert broadcast_shapes((2, 3), (1, 3)) == (2, 3)

    def test_outer_broadcast(self) -> None:
        assert broadcast_shapes((2, 1), (1, 4)) == (2, 4)

    def test_scalar_shape(self) -> None:
        assert broadcast_shapes((), (2, 3)) == (2, 3)

    def test_incompatible_shapes_raise(self) -> None:
        with pytest.raises(BroadcastError, match=r"\(2, 3\).*\(2, 4\)"):
            broadcast_shapes((2, 3), (2, 4))


BROADCAST_CASES = [
    ((3,), (3,)),
    ((2, 3), (3,)),
    ((2, 3), (1, 3)),
    ((2, 1), (1, 4)),
    ((), (2, 3)),
    ((1, 3), (2, 1)),
    ((4, 1, 3), (3,)),
]


@pytest.mark.parametrize("shape_a,shape_b", BROADCAST_CASES)
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_broadcast_arithmetic_numpy_parity(
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    op: str,
) -> None:
    rng = np.random.default_rng(0)
    array_a = rng.integers(1, 5, size=shape_a).astype(np.float64)
    array_b = rng.integers(1, 5, size=shape_b).astype(np.float64)
    tensor_a = Tensor(array_a, dtype=Dtype.float64)
    tensor_b = Tensor(array_b, dtype=Dtype.float64)
    numpy_ops = {
        "add": array_a + array_b,
        "sub": array_a - array_b,
        "mul": array_a * array_b,
        "div": array_a / array_b,
    }
    titan_ops = {
        "add": tensor_a + tensor_b,
        "sub": tensor_a - tensor_b,
        "mul": tensor_a * tensor_b,
        "div": tensor_a / tensor_b,
    }
    result = titan_ops[op]
    expected = numpy_ops[op]
    assert_tensor_allclose(result, expected)
    assert_result_metadata(result, expected, Dtype.float64)


class TestBroadcastOperators:
    def test_vector_plus_vector(self) -> None:
        result = Tensor([1, 2, 3], dtype=Dtype.int32) + Tensor([4, 5, 6], dtype=Dtype.int32)
        expected = np.array([1, 2, 3], dtype=np.int32) + np.array([4, 5, 6], dtype=np.int32)
        assert_tensor_equal(result, expected)
        assert result.shape == (3,)

    def test_matrix_plus_vector(self) -> None:
        matrix = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        vector = Tensor([10, 20, 30], dtype=Dtype.int32)
        result = matrix + vector
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32) + np.array(
            [10, 20, 30], dtype=np.int32
        )
        assert_tensor_equal(result, expected)
        assert_result_metadata(result, expected, Dtype.int32)

    def test_matrix_plus_row(self) -> None:
        matrix = Tensor([[1, 2, 3], [4, 5, 6]], dtype=Dtype.int32)
        row = Tensor([[10, 20, 30]], dtype=Dtype.int32)
        result = matrix + row
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32) + np.array(
            [[10, 20, 30]], dtype=np.int32
        )
        assert_tensor_equal(result, expected)

    def test_column_plus_row(self) -> None:
        column = Tensor([[1], [2]], dtype=Dtype.int32)
        row = Tensor([[10, 20, 30, 40]], dtype=Dtype.int32)
        result = column + row
        expected = np.array([[1], [2]], dtype=np.int32) + np.array(
            [[10, 20, 30, 40]], dtype=np.int32
        )
        assert_tensor_equal(result, expected)
        assert result.shape == (2, 4)

    def test_scalar_plus_tensor(self) -> None:
        tensor = Tensor([[1, 2], [3, 4]], dtype=Dtype.int32)
        result = 5 + tensor
        expected = 5 + np.array([[1, 2], [3, 4]], dtype=np.int32)
        assert_tensor_equal(result, expected)
        assert result.shape == (2, 2)
        assert result.device == Device.cpu()

    def test_incompatible_shapes_raise_broadcast_error(self) -> None:
        left = Tensor([[1, 2, 3], [4, 5, 6]])
        right = Tensor([[1, 2], [3, 4]])
        with pytest.raises(BroadcastError, match="Incompatible broadcast shapes"):
            _ = left + right
        with pytest.raises(BroadcastError):
            _ = left * right
        with pytest.raises(BroadcastError):
            _ = left / Tensor([1, 2])
