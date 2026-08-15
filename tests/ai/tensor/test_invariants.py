"""Invariant tests for Tensor operations (Day 6 hardening)."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import assert_tensor_allclose, assert_tensor_equal, tensor_numpy_array
from titan_ai.tensor import Dtype, Tensor
from titan_ai.tensor.operations.broadcast import broadcast_shapes


class TestArithmeticInvariants:
    def test_add_zero_preserves_values(self) -> None:
        tensor = Tensor([1, 2, 3], dtype=Dtype.int32)
        result = tensor + 0
        assert_tensor_equal(result, tensor_numpy_array(tensor))

    def test_mul_one_preserves_values(self) -> None:
        tensor = Tensor([1.5, -2.0, 0.0], dtype=Dtype.float64)
        result = tensor * 1
        assert_tensor_allclose(result, tensor_numpy_array(tensor))


class TestShapeInvariants:
    def test_reshape_preserves_size(self) -> None:
        original = Tensor(np.arange(24, dtype=np.int32).reshape(2, 3, 4))
        reshaped = original.reshape(4, 6)
        assert reshaped.size == original.size
        assert reshaped.shape == (4, 6)

    def test_default_transpose_reverses_shape(self) -> None:
        tensor = Tensor(np.arange(24).reshape(2, 3, 4))
        transposed = tensor.transpose()
        assert transposed.shape == tuple(reversed(tensor.shape))
        assert transposed.size == tensor.size

    def test_flatten_preserves_size(self) -> None:
        tensor = Tensor(np.arange(24).reshape(2, 3, 4))
        flat = tensor.flatten()
        assert flat.size == tensor.size
        assert flat.ndim == 1

    def test_unsqueeze_then_squeeze_restores_shape(self) -> None:
        tensor = Tensor(np.arange(6).reshape(2, 3), dtype=Dtype.int32)
        restored = tensor.unsqueeze(1).squeeze(axis=1)
        assert restored.shape == tensor.shape
        assert_tensor_equal(restored, tensor_numpy_array(tensor))


class TestReductionInvariants:
    def test_keepdims_preserves_rank(self) -> None:
        tensor = Tensor(np.arange(24, dtype=np.int32).reshape(2, 3, 4))
        for axis in (0, 1, -1):
            result = tensor.sum(axis=axis, keepdims=True)
            assert result.ndim == tensor.ndim
            assert result.shape[axis if axis >= 0 else axis + tensor.ndim] == 1


class TestBroadcastInvariants:
    @pytest.mark.parametrize(
        ("shape_a", "shape_b"),
        [
            ((), (3,)),
            ((1,), (3,)),
            ((2, 3), (1, 3)),
            ((2, 1), (1, 4)),
            ((2, 1, 4), (1, 3, 1)),
            ((1, 1), (2, 3)),
        ],
    )
    def test_output_shape_matches_broadcast_rules(
        self,
        shape_a: tuple[int, ...],
        shape_b: tuple[int, ...],
    ) -> None:
        left = Tensor.ones(shape_a, dtype=Dtype.float64)
        right = Tensor.full(shape_b, 2.0, dtype=Dtype.float64)
        result = left + right
        expected_shape = broadcast_shapes(shape_a, shape_b)
        assert result.shape == expected_shape
        expected = np.ones(shape_a) + np.full(shape_b, 2.0)
        assert_tensor_allclose(result, expected)
