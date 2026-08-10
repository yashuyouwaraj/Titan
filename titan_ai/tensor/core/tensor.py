"""Core Tensor abstraction."""

from typing import Any

import numpy as np

from titan_ai.tensor.backend.base import TensorBackend
from titan_ai.tensor.backend.numpy_backend import NumpyBackend
from titan_ai.tensor.devices.devices import Device
from titan_ai.tensor.dtypes.dtypes import Dtype
from titan_ai.tensor.exceptions.errors import (
    TensorConstructionError,
    UnsupportedDeviceError,
    UnsupportedDtypeError,
)

ArrayLike = Any


def _create_backend(
    data: ArrayLike,
    dtype: Dtype,
    device: Device,
    copy: bool,
) -> TensorBackend:
    if device.type == "cpu":
        return NumpyBackend.from_data(data, dtype=dtype, device=device, copy=copy)
    raise UnsupportedDeviceError(
        f"Unsupported device {device!r}. Only CPU execution is available in Week 1."
    )


def _resolve_dtype(data: ArrayLike, dtype: Dtype | None) -> Dtype:
    if dtype is not None:
        return dtype
    try:
        if isinstance(data, np.ndarray):
            return Dtype.infer_from_array(data)
        preview_array = np.asarray(data)
        return Dtype.infer_from_array(preview_array)
    except UnsupportedDtypeError:
        raise
    except (ValueError, TypeError) as exc:
        raise TensorConstructionError(f"Failed to infer dtype from data: {exc}") from exc


class Tensor:
    """Owned tensor abstraction with explicit metadata and backend storage.

    Metadata properties (``shape``, ``dtype``, ``device``, ``ndim``, ``size``)
    are read-only. They are derived from backend storage and must not be
    assigned directly by users.
    """

    def __init__(
        self,
        data: ArrayLike,
        *,
        dtype: Dtype | None = None,
        device: Device | None = None,
        copy: bool = True,
    ) -> None:
        resolved_device = device if device is not None else Device.cpu()
        resolved_dtype = _resolve_dtype(data, dtype)
        self._backend = _create_backend(
            data=data,
            dtype=resolved_dtype,
            device=resolved_device,
            copy=copy,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor shape."""
        return self._backend.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self._backend.ndim

    @property
    def dtype(self) -> Dtype:
        """Return the tensor element dtype."""
        return self._backend.dtype

    @property
    def device(self) -> Device:
        """Return the tensor device."""
        return self._backend.device

    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return self._backend.size

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype.name}, device={self.device!r})"
