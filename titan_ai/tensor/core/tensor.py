"""Core Tensor abstraction."""

from typing import Any

import numpy as np

from titan_ai.tensor.backend.base import TensorBackend
from titan_ai.tensor.backend.numpy_backend import NumpyBackend
from titan_ai.tensor.devices.devices import Device
from titan_ai.tensor.dtypes.dtypes import Dtype
from titan_ai.tensor.exceptions.errors import (
    TensorConstructionError,
    TensorValidationError,
    UnsupportedDeviceError,
    UnsupportedDtypeError,
)
from titan_ai.tensor.operations.shape import normalize_shape, resolve_inferred_shape

ArrayLike = Any

_DEFAULT_FACTORY_DTYPE = Dtype.float64


def _resolve_device(device: Device | None) -> Device:
    return device if device is not None else Device.cpu()


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


def _create_factory_backend(
    factory: str,
    shape: tuple[int, ...],
    dtype: Dtype,
    device: Device,
    value: int | float | bool | None = None,
    arange_args: tuple[float, float | None, float] | None = None,
) -> TensorBackend:
    if device.type != "cpu":
        raise UnsupportedDeviceError(
            f"Unsupported device {device!r}. Only CPU execution is available in Week 1."
        )
    if factory == "zeros":
        return NumpyBackend.zeros(shape, dtype=dtype, device=device)
    if factory == "ones":
        return NumpyBackend.ones(shape, dtype=dtype, device=device)
    if factory == "empty":
        return NumpyBackend.empty(shape, dtype=dtype, device=device)
    if factory == "full" and value is not None:
        return NumpyBackend.full(shape, value, dtype=dtype, device=device)
    if factory == "arange" and arange_args is not None:
        start, stop, step = arange_args
        return NumpyBackend.arange(start, stop, step, dtype=dtype, device=device)
    raise TensorConstructionError(f"Unknown factory operation: {factory}")


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


def _infer_arange_dtype(
    start: int | float,
    stop: int | float | None,
    step: int | float,
    dtype: Dtype | None,
) -> Dtype:
    if dtype is not None:
        return dtype
    preview_stop = stop if stop is not None else start
    preview_start = 0 if stop is None else start
    preview = np.arange(preview_start, preview_stop, step)
    return Dtype.infer_from_array(preview)


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
        resolved_device = _resolve_device(device)
        resolved_dtype = _resolve_dtype(data, dtype)
        self._backend = _create_backend(
            data=data,
            dtype=resolved_dtype,
            device=resolved_device,
            copy=copy,
        )

    @classmethod
    def _from_backend(cls, backend: TensorBackend) -> "Tensor":
        tensor = cls.__new__(cls)
        tensor._backend = backend
        return tensor

    @staticmethod
    def zeros(
        shape: int | tuple[int, ...] | list[int],
        *,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> "Tensor":
        """Create a tensor filled with zeros."""
        normalized_shape = normalize_shape(shape)
        resolved_dtype = dtype if dtype is not None else _DEFAULT_FACTORY_DTYPE
        resolved_device = _resolve_device(device)
        backend = _create_factory_backend(
            "zeros",
            normalized_shape,
            resolved_dtype,
            resolved_device,
        )
        return Tensor._from_backend(backend)

    @staticmethod
    def ones(
        shape: int | tuple[int, ...] | list[int],
        *,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> "Tensor":
        """Create a tensor filled with ones."""
        normalized_shape = normalize_shape(shape)
        resolved_dtype = dtype if dtype is not None else _DEFAULT_FACTORY_DTYPE
        resolved_device = _resolve_device(device)
        backend = _create_factory_backend(
            "ones",
            normalized_shape,
            resolved_dtype,
            resolved_device,
        )
        return Tensor._from_backend(backend)

    @staticmethod
    def empty(
        shape: int | tuple[int, ...] | list[int],
        *,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> "Tensor":
        """Create a tensor with uninitialized storage."""
        normalized_shape = normalize_shape(shape)
        resolved_dtype = dtype if dtype is not None else _DEFAULT_FACTORY_DTYPE
        resolved_device = _resolve_device(device)
        backend = _create_factory_backend(
            "empty",
            normalized_shape,
            resolved_dtype,
            resolved_device,
        )
        return Tensor._from_backend(backend)

    @staticmethod
    def full(
        shape: int | tuple[int, ...] | list[int],
        value: int | float | bool,
        *,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> "Tensor":
        """Create a tensor filled with a constant value."""
        normalized_shape = normalize_shape(shape)
        resolved_dtype = dtype if dtype is not None else _resolve_dtype(value, None)
        resolved_device = _resolve_device(device)
        backend = _create_factory_backend(
            "full",
            normalized_shape,
            resolved_dtype,
            resolved_device,
            value=value,
        )
        return Tensor._from_backend(backend)

    @staticmethod
    def arange(
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        *,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> "Tensor":
        """Create a tensor containing an arithmetic range."""
        if step == 0:
            raise TensorConstructionError("arange step must be non-zero.")
        resolved_dtype = _infer_arange_dtype(start, stop, step, dtype)
        resolved_device = _resolve_device(device)
        backend = _create_factory_backend(
            "arange",
            (),
            resolved_dtype,
            resolved_device,
            arange_args=(start, stop, step),
        )
        return Tensor._from_backend(backend)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        dtype: Dtype | None = None,
        device: Device | None = None,
        copy: bool = True,
    ) -> "Tensor":
        """Create a tensor from a NumPy array with explicit copy semantics."""
        if not isinstance(array, np.ndarray):
            raise TensorConstructionError("from_numpy requires a NumPy ndarray input.")
        return Tensor(array, dtype=dtype, device=device, copy=copy)

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

    @property
    def T(self) -> "Tensor":
        """Return the transpose of a 2D tensor."""
        if self.ndim != 2:
            raise TensorValidationError("T property is only defined for 2D tensors.")
        return self.transpose(1, 0)

    def reshape(self, *shape: int) -> "Tensor":
        """Return a tensor with the same data arranged in a new shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = normalize_shape(shape[0])
        else:
            target_shape = normalize_shape(shape)
        resolved_shape = resolve_inferred_shape(self.size, target_shape)
        return Tensor._from_backend(self._backend.reshape(resolved_shape))

    def transpose(self, *axes: int) -> "Tensor":
        """Return a tensor with permuted axes."""
        axis_tuple = tuple(axes) if axes else None
        return Tensor._from_backend(self._backend.transpose_axes(axis_tuple))

    def flatten(self) -> "Tensor":
        """Return a 1D copy of the tensor."""
        return Tensor._from_backend(self._backend.flatten())

    def squeeze(self, axis: int | None = None) -> "Tensor":
        """Return a tensor with singleton dimensions removed."""
        return Tensor._from_backend(self._backend.squeeze(axis=axis))

    def unsqueeze(self, axis: int) -> "Tensor":
        """Insert a singleton dimension at the given axis."""
        return Tensor._from_backend(self._backend.unsqueeze(axis=axis))

    def numpy(self, copy: bool = True) -> np.ndarray:
        """Return the tensor data as a NumPy array.

        By default this returns a copy. Pass ``copy=False`` to obtain a view
        that shares storage with the tensor when the backend supports views.
        """
        backend = self._backend
        if not isinstance(backend, NumpyBackend):
            raise TensorValidationError("numpy() is only supported for CPU tensors in Week 1.")
        array = backend.numpy_array
        if copy:
            return array.copy()
        return array

    def __getitem__(self, key: Any) -> "Tensor | int | float | bool":
        result = self._backend.get_item(key)
        if isinstance(result, TensorBackend):
            return Tensor._from_backend(result)
        return result

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype.name}, device={self.device!r})"
