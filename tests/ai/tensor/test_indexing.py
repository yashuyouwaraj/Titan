"""Tests for Tensor indexing and slicing."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import assert_tensor_equal, tensor_numpy_array
from titan_ai.tensor import Tensor
from titan_ai.tensor.exceptions.errors import TensorIndexError


class TestIntegerIndexing:
    def test_first_element(self) -> None:
        tensor = Tensor([10, 20, 30])
        assert tensor[0] == 10

    def test_negative_index(self) -> None:
        tensor = Tensor([10, 20, 30])
        assert tensor[-1] == 30

    def test_multidimensional_index(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        assert tensor[1, 2] == 6


class TestSliceIndexing:
    def test_simple_slice(self) -> None:
        tensor = Tensor([1, 2, 3, 4, 5])[1:3]
        assert isinstance(tensor, Tensor)
        assert tensor.shape == (2,)
        assert_tensor_equal(tensor, np.array([1, 2, 3, 4, 5])[1:3])

    def test_slice_with_step(self) -> None:
        tensor = Tensor([1, 2, 3, 4, 5])[::2]
        assert_tensor_equal(tensor, np.array([1, 2, 3, 4, 5])[::2])

    def test_multidimensional_slice(self) -> None:
        array = np.arange(12).reshape(3, 4)
        tensor = Tensor(array)[1:, :2]
        assert isinstance(tensor, Tensor)
        assert tensor.shape == (2, 2)
        assert_tensor_equal(tensor, array[1:, :2])

    def test_column_slice(self) -> None:
        array = np.arange(12).reshape(3, 4)
        tensor = Tensor(array)[:, 0]
        assert isinstance(tensor, Tensor)
        assert_tensor_equal(tensor, array[:, 0])


class TestScalarTensorIndexing:
    def test_scalar_tensor_returns_scalar(self) -> None:
        tensor = Tensor(7)
        assert tensor[()] == 7


class TestIndexingValidation:
    def test_out_of_bounds_raises(self) -> None:
        tensor = Tensor([1, 2, 3])
        with pytest.raises(TensorIndexError):
            tensor[5]

    def test_invalid_index_type_raises(self) -> None:
        tensor = Tensor([1, 2, 3])
        with pytest.raises(TensorIndexError):
            tensor["invalid"]


class TestSliceMemorySemantics:
    def test_slice_returns_view(self) -> None:
        tensor = Tensor([1, 2, 3, 4, 5])
        sliced = tensor[1:4]
        assert isinstance(sliced, Tensor)
        tensor_numpy_array(tensor)[2] = 99
        assert tensor_numpy_array(sliced)[1] == 99


class TestNumpyConversion:
    def test_numpy_returns_copy_by_default(self) -> None:
        tensor = Tensor([1, 2, 3])
        array = tensor.numpy()
        array[0] = 99
        assert tensor_numpy_array(tensor)[0] == 1

    def test_numpy_view_when_copy_false(self) -> None:
        tensor = Tensor([1, 2, 3])
        array = tensor.numpy(copy=False)
        array[0] = 99
        assert tensor_numpy_array(tensor)[0] == 99
