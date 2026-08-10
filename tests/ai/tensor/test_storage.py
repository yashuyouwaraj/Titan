"""Tests for storage ownership semantics."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import tensor_numpy_array
from titan_ai.tensor import Dtype, Tensor
from titan_ai.tensor.exceptions.errors import TensorValidationError


class TestStorageOwnership:
    def test_default_copy_from_numpy_isolated(self) -> None:
        array = np.array([1, 2, 3], dtype=np.int32)
        tensor = Tensor(array)
        array[0] = 99
        assert tensor_numpy_array(tensor)[0] == 1

    def test_explicit_copy_from_numpy_isolated(self) -> None:
        array = np.array([1, 2, 3], dtype=np.int32)
        tensor = Tensor(array, copy=True)
        array[0] = 99
        assert tensor_numpy_array(tensor)[0] == 1

    def test_zero_copy_shares_storage(self) -> None:
        array = np.array([1, 2, 3], dtype=np.int32)
        tensor = Tensor(array, copy=False)
        array[0] = 99
        assert tensor_numpy_array(tensor)[0] == 99

    def test_list_input_always_creates_new_storage(self) -> None:
        data = [1, 2, 3]
        tensor = Tensor(data)
        data[0] = 99
        assert tensor_numpy_array(tensor)[0] == 1

    def test_zero_copy_requires_compatible_dtype(self) -> None:
        array = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(TensorValidationError):
            Tensor(array, dtype=Dtype.float32, copy=False)
