"""Tensor Library exception hierarchy."""

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
