"""Backend contract for tensor storage."""

from abc import ABC, abstractmethod
from typing import Any

from titan_ai.tensor.devices.devices import Device
from titan_ai.tensor.dtypes.dtypes import Dtype


class TensorBackend(ABC):
    """Abstract storage backend for a tensor.

    Concrete backends (NumPy CPU today, CUDA in future) implement this
    contract so the public ``Tensor`` API can remain stable.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the tensor shape."""

    @property
    @abstractmethod
    def dtype(self) -> Dtype:
        """Return the tensor element dtype."""

    @property
    @abstractmethod
    def device(self) -> Device:
        """Return the tensor device."""

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the total number of elements."""
        if not self.shape:
            return 1
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    @abstractmethod
    def reshape(self, shape: tuple[int, ...]) -> "TensorBackend":
        """Return a tensor with the requested shape."""

    @abstractmethod
    def transpose_axes(self, axes: tuple[int, ...] | None = None) -> "TensorBackend":
        """Return a tensor with permuted axes."""

    @abstractmethod
    def flatten(self) -> "TensorBackend":
        """Return a 1D copy of the tensor."""

    @abstractmethod
    def squeeze(self, axis: int | None = None) -> "TensorBackend":
        """Return a tensor with singleton dimensions removed."""

    @abstractmethod
    def unsqueeze(self, axis: int) -> "TensorBackend":
        """Return a tensor with a singleton dimension inserted."""

    @abstractmethod
    def get_item(self, key: Any) -> "TensorBackend | int | float | bool":
        """Return an indexed sub-tensor or scalar value."""

    @abstractmethod
    def add(self, other: "TensorBackend", dtype: Dtype) -> "TensorBackend":
        """Return element-wise addition with ``other`` using ``dtype``."""

    @abstractmethod
    def subtract(self, other: "TensorBackend", dtype: Dtype) -> "TensorBackend":
        """Return element-wise subtraction with ``other`` using ``dtype``."""

    @abstractmethod
    def multiply(self, other: "TensorBackend", dtype: Dtype) -> "TensorBackend":
        """Return element-wise multiplication with ``other`` using ``dtype``."""

    @abstractmethod
    def true_divide(self, other: "TensorBackend", dtype: Dtype) -> "TensorBackend":
        """Return element-wise true division with ``other`` using ``dtype``."""

    @abstractmethod
    def negate(self, dtype: Dtype) -> "TensorBackend":
        """Return element-wise negation using ``dtype``."""

    @abstractmethod
    def matmul(self, other: "TensorBackend", dtype: Dtype) -> "TensorBackend":
        """Return matrix multiplication with ``other`` using ``dtype``."""
