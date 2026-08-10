"""NumPy CPU storage backend for Titan tensors."""

from typing import Any

import numpy as np

from titan_ai.tensor.backend.base import TensorBackend
from titan_ai.tensor.devices.devices import Device
from titan_ai.tensor.dtypes.dtypes import Dtype
from titan_ai.tensor.exceptions.errors import (
    InvalidAxisError,
    InvalidShapeError,
    TensorConstructionError,
    TensorIndexError,
    TensorValidationError,
)


class NumpyBackend(TensorBackend):
    """CPU tensor storage backed by a NumPy ndarray."""

    def __init__(self, array: np.ndarray, device: Device) -> None:
        if device.type != "cpu":
            raise TensorValidationError(f"NumpyBackend requires a CPU device, got {device!r}.")
        if not isinstance(array, np.ndarray):
            raise TensorValidationError("NumpyBackend storage must be a NumPy ndarray.")
        self._array = array
        self._device = device
        self._dtype = Dtype.infer_from_array(array)

    @classmethod
    def from_array(cls, array: np.ndarray, device: Device) -> "NumpyBackend":
        """Wrap an existing NumPy array without copying."""
        return cls(array, device)

    @classmethod
    def from_data(
        cls,
        data: Any,
        dtype: Dtype,
        device: Device,
        copy: bool,
    ) -> "NumpyBackend":
        """Create a backend from array-like input."""
        np_dtype = dtype.to_numpy()
        try:
            if isinstance(data, np.ndarray):
                if copy:
                    array = np.array(data, dtype=np_dtype, copy=True)
                else:
                    if data.dtype != np_dtype:
                        raise TensorValidationError(
                            f"Zero-copy construction requires dtype {dtype.name}, "
                            f"but array has dtype {data.dtype!r}."
                        )
                    array = data
            else:
                array = np.asarray(data, dtype=np_dtype)
        except (ValueError, TypeError) as exc:
            raise TensorConstructionError(f"Failed to construct tensor from data: {exc}") from exc

        if array.dtype != np_dtype:
            raise TensorValidationError(
                f"Internal dtype mismatch: expected {np_dtype!r}, got {array.dtype!r}."
            )

        return cls(array, device)

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        dtype: Dtype,
        device: Device,
    ) -> "NumpyBackend":
        """Create a zero-filled backend."""
        array = np.zeros(shape, dtype=dtype.to_numpy())
        return cls(array, device)

    @classmethod
    def ones(
        cls,
        shape: tuple[int, ...],
        dtype: Dtype,
        device: Device,
    ) -> "NumpyBackend":
        """Create a one-filled backend."""
        array = np.ones(shape, dtype=dtype.to_numpy())
        return cls(array, device)

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        dtype: Dtype,
        device: Device,
    ) -> "NumpyBackend":
        """Create uninitialized backend storage."""
        array = np.empty(shape, dtype=dtype.to_numpy())
        return cls(array, device)

    @classmethod
    def full(
        cls,
        shape: tuple[int, ...],
        value: int | float | bool,
        dtype: Dtype,
        device: Device,
    ) -> "NumpyBackend":
        """Create a backend filled with a constant value."""
        array = np.full(shape, value, dtype=dtype.to_numpy())
        return cls(array, device)

    @classmethod
    def arange(
        cls,
        start: int | float,
        stop: int | float | None,
        step: int | float,
        dtype: Dtype,
        device: Device,
    ) -> "NumpyBackend":
        """Create a backend from an arithmetic range."""
        try:
            if stop is None:
                array = np.arange(start, step=step, dtype=dtype.to_numpy())
            else:
                array = np.arange(start, stop, step, dtype=dtype.to_numpy())
        except (ValueError, TypeError) as exc:
            raise TensorConstructionError(f"Invalid arange arguments: {exc}") from exc
        return cls(array, device)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self._array.shape)

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    @property
    def size(self) -> int:
        return int(self._array.size)

    @property
    def numpy_array(self) -> np.ndarray:
        """Return the underlying NumPy array (internal use within the tensor package)."""
        return self._array

    def reshape(self, shape: tuple[int, ...]) -> "NumpyBackend":
        try:
            array = self._array.reshape(shape)
        except ValueError as exc:
            raise InvalidShapeError(f"Cannot reshape to shape {shape}: {exc}") from exc
        return NumpyBackend.from_array(array, self._device)

    def transpose_axes(self, axes: tuple[int, ...] | None = None) -> "NumpyBackend":
        try:
            array = self._array.transpose() if axes is None else self._array.transpose(axes)
        except ValueError as exc:
            raise InvalidAxisError(f"Invalid transpose axes {axes}: {exc}") from exc
        return NumpyBackend.from_array(array, self._device)

    def flatten(self) -> "NumpyBackend":
        array = self._array.flatten()
        return NumpyBackend.from_array(array, self._device)

    def squeeze(self, axis: int | None = None) -> "NumpyBackend":
        try:
            array = np.squeeze(self._array, axis=axis)
        except ValueError as exc:
            raise InvalidAxisError(f"Invalid squeeze axis {axis}: {exc}") from exc
        return NumpyBackend.from_array(array, self._device)

    def unsqueeze(self, axis: int) -> "NumpyBackend":
        try:
            array = np.expand_dims(self._array, axis=axis)
        except ValueError as exc:
            raise InvalidAxisError(f"Invalid unsqueeze axis {axis}: {exc}") from exc
        return NumpyBackend.from_array(array, self._device)

    def get_item(self, key: Any) -> "NumpyBackend | int | float | bool":
        try:
            result = self._array[key]
        except IndexError as exc:
            raise TensorIndexError(f"Invalid index {key!r}: {exc}") from exc
        except TypeError as exc:
            raise TensorIndexError(f"Invalid index {key!r}: {exc}") from exc

        if isinstance(result, np.ndarray):
            return NumpyBackend.from_array(result, self._device)
        if isinstance(result, np.generic):
            return result.item()
        return result
