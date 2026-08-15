"""Tests for Device abstraction."""

import pytest

from titan_ai.tensor import Device, Tensor
from titan_ai.tensor.exceptions.errors import UnsupportedDeviceError


class TestDevice:
    def test_cpu_constructor(self) -> None:
        device = Device("cpu")
        assert device.type == "cpu"
        assert device == Device.cpu()

    def test_cpu_class_method(self) -> None:
        assert Device.cpu().type == "cpu"

    def test_device_equality_with_string(self) -> None:
        assert Device("cpu") == "cpu"

    def test_tensor_defaults_to_cpu(self) -> None:
        tensor = Tensor([1, 2, 3])
        assert tensor.device == Device.cpu()

    def test_tensor_with_explicit_cpu(self) -> None:
        tensor = Tensor([1, 2, 3], device=Device("cpu"))
        assert tensor.device == Device.cpu()

    def test_tensor_accepts_cpu_string(self) -> None:
        tensor = Tensor([1, 2, 3], device="cpu")
        assert tensor.device == Device.cpu()
        zeros = Tensor.zeros((2,), device="cpu")
        assert zeros.device == Device.cpu()

    def test_cuda_device_raises(self) -> None:
        with pytest.raises(UnsupportedDeviceError, match="CUDA"):
            Device("cuda")

    def test_unknown_device_raises(self) -> None:
        with pytest.raises(UnsupportedDeviceError, match="Unsupported device"):
            Device("tpu")

    def test_invalid_device_string_on_tensor_raises_domain_error(self) -> None:
        with pytest.raises(UnsupportedDeviceError):
            Tensor([1, 2, 3], device="tpu")

    def test_tensor_with_cuda_device_raises(self) -> None:
        with pytest.raises(UnsupportedDeviceError, match="CUDA"):
            Tensor([1, 2, 3], device=Device("cuda"))

    def test_repr(self) -> None:
        assert repr(Device.cpu()) == "Device('cpu')"
