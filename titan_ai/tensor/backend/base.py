"""Backend contract for tensor storage."""

from abc import ABC, abstractmethod

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
