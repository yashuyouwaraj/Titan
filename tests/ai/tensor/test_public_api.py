"""Tests for public API imports."""

from titan_ai.tensor import (
    BroadcastError,
    Device,
    DeviceMismatchError,
    Dtype,
    Tensor,
    TensorConstructionError,
    TensorValidationError,
    TitanTensorError,
    UnsupportedDeviceError,
    UnsupportedDtypeError,
    UnsupportedOperandError,
)


class TestPublicApi:
    def test_tensor_import(self) -> None:
        assert Tensor is not None

    def test_dtype_import(self) -> None:
        assert Dtype.float32.name == "float32"

    def test_device_import(self) -> None:
        assert Device.cpu().type == "cpu"

    def test_exception_imports(self) -> None:
        assert issubclass(UnsupportedDeviceError, TitanTensorError)
        assert issubclass(UnsupportedDtypeError, TitanTensorError)
        assert issubclass(TensorConstructionError, TitanTensorError)
        assert issubclass(TensorValidationError, TitanTensorError)
        assert issubclass(BroadcastError, TitanTensorError)
        assert issubclass(DeviceMismatchError, TitanTensorError)
        assert issubclass(UnsupportedOperandError, TitanTensorError)
