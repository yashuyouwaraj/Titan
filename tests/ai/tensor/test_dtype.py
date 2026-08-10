"""Tests for Dtype abstraction."""

import numpy as np
import pytest

from tests.ai.tensor.conftest import tensor_numpy_array
from titan_ai.tensor import Dtype, Tensor
from titan_ai.tensor.exceptions.errors import UnsupportedDtypeError


class TestDtype:
    def test_float32(self) -> None:
        tensor = Tensor([1.0, 2.0], dtype=Dtype.float32)
        assert tensor.dtype == Dtype.float32
        assert tensor_numpy_array(tensor).dtype == np.float32

    def test_float64(self) -> None:
        tensor = Tensor([1.0, 2.0], dtype=Dtype.float64)
        assert tensor.dtype == Dtype.float64
        assert tensor_numpy_array(tensor).dtype == np.float64

    def test_int32(self) -> None:
        tensor = Tensor([1, 2], dtype=Dtype.int32)
        assert tensor.dtype == Dtype.int32
        assert tensor_numpy_array(tensor).dtype == np.int32

    def test_int64(self) -> None:
        tensor = Tensor([1, 2], dtype=Dtype.int64)
        assert tensor.dtype == Dtype.int64
        assert tensor_numpy_array(tensor).dtype == np.int64

    def test_bool(self) -> None:
        tensor = Tensor([True, False], dtype=Dtype.bool)
        assert tensor.dtype == Dtype.bool
        assert tensor_numpy_array(tensor).dtype == np.bool_

    def test_to_numpy_roundtrip(self) -> None:
        for member in Dtype:
            assert Dtype.from_numpy(member.to_numpy()) == member

    def test_infer_from_unsupported_numpy_dtype_raises(self) -> None:
        array = np.array([1, 2], dtype=np.float16)
        with pytest.raises(UnsupportedDtypeError):
            Dtype.from_numpy(array.dtype)

    def test_construct_with_unsupported_inferred_dtype_raises(self) -> None:
        array = np.array([1, 2], dtype=np.uint8)
        with pytest.raises(UnsupportedDtypeError):
            Tensor(array)
