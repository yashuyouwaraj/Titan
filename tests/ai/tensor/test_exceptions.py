"""Tests for tensor exception hierarchy."""

import numpy as np
import pytest

from titan_ai.tensor import Device, Dtype, Tensor
from titan_ai.tensor.exceptions.errors import (
    TensorConstructionError,
    TensorValidationError,
    TitanTensorError,
    UnsupportedDeviceError,
    UnsupportedDtypeError,
)


class TestExceptions:
    def test_unsupported_device_is_titan_tensor_error(self) -> None:
        with pytest.raises(UnsupportedDeviceError) as exc_info:
            Device("cuda")
        assert isinstance(exc_info.value, TitanTensorError)

    def test_unsupported_dtype_is_titan_tensor_error(self) -> None:
        with pytest.raises(UnsupportedDtypeError) as exc_info:
            Dtype.from_numpy(np.dtype(np.float16))
        assert isinstance(exc_info.value, TitanTensorError)

    def test_construction_error_on_invalid_data(self) -> None:
        with pytest.raises(TensorConstructionError) as exc_info:
            Tensor([[1, 2], [3]])
        assert isinstance(exc_info.value, TitanTensorError)

    def test_validation_error_on_zero_copy_dtype_mismatch(self) -> None:
        array = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(TensorValidationError):
            Tensor(array, dtype=Dtype.float64, copy=False)
