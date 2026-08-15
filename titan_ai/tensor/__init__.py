"""Titan AI Tensor Library — foundational numerical tensor abstraction."""

from titan_ai.tensor.core.tensor import Tensor
from titan_ai.tensor.devices.devices import Device
from titan_ai.tensor.dtypes.dtypes import Dtype
from titan_ai.tensor.exceptions.errors import (
    BroadcastError,
    DeviceMismatchError,
    InvalidAxisError,
    InvalidShapeError,
    TensorConstructionError,
    TensorIndexError,
    TensorValidationError,
    TitanTensorError,
    UnsupportedDeviceError,
    UnsupportedDtypeError,
    UnsupportedOperandError,
)

__all__ = [
    "Tensor",
    "Dtype",
    "Device",
    "TitanTensorError",
    "TensorConstructionError",
    "TensorValidationError",
    "UnsupportedDeviceError",
    "UnsupportedDtypeError",
    "InvalidShapeError",
    "InvalidAxisError",
    "TensorIndexError",
    "BroadcastError",
    "DeviceMismatchError",
    "UnsupportedOperandError",
]
