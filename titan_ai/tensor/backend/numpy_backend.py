"""NumPy CPU storage backend for Titan tensors."""

from typing import Any

import numpy as np

from titan_ai.tensor.backend.base import TensorBackend
from titan_ai.tensor.devices.devices import Device
from titan_ai.tensor.dtypes.dtypes import Dtype
from titan_ai.tensor.exceptions.errors import (
    TensorConstructionError,
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
