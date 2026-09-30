"""Core Tensor abstraction."""

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from titan_ai.tensor.autograd.graph import AutogradNode

ArrayLike = Any

_DEFAULT_FACTORY_DTYPE = Dtype.float64


def _resolve_device(device: Device | str | None) -> Device:
    if device is None:
        return Device.cpu()
    if isinstance(device, Device):
        return device
    if isinstance(device, str):
        return Device(device)
    raise UnsupportedDeviceError(
        f"Unsupported device {device!r}. Provide Device.cpu() or the string 'cpu'."
    )


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
        device: Device | str | None = None,
        copy: bool = True,
        requires_grad: bool = False,
    ) -> None:
        resolved_device = _resolve_device(device)
        resolved_dtype = _resolve_dtype(data, dtype)
        self._backend = _create_backend(
            data=data,
            dtype=resolved_dtype,
            device=resolved_device,
            copy=copy,
        )
        self._requires_grad: bool = requires_grad
        self._grad: "Tensor | None" = None
        self._grad_fn: "AutogradNode | None" = None
        self._is_leaf: bool = True

    @classmethod
    def _from_backend(cls, backend: TensorBackend, requires_grad: bool = False) -> "Tensor":
        tensor = cls.__new__(cls)
        tensor._backend = backend
        tensor._requires_grad = requires_grad
        tensor._grad = None
        tensor._grad_fn = None
        tensor._is_leaf = True
        return tensor

    @staticmethod
    def zeros(
        shape: int | tuple[int, ...] | list[int],
        *,
        dtype: Dtype | None = None,
        device: Device | str | None = None,
        requires_grad: bool = False,
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
        return Tensor._from_backend(backend, requires_grad=requires_grad)

    @staticmethod
    def ones(
        shape: int | tuple[int, ...] | list[int],
        *,
        dtype: Dtype | None = None,
        device: Device | str | None = None,
        requires_grad: bool = False,
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
        return Tensor._from_backend(backend, requires_grad=requires_grad)

    @staticmethod
    def empty(
        shape: int | tuple[int, ...] | list[int],
        *,
        dtype: Dtype | None = None,
        device: Device | str | None = None,
        requires_grad: bool = False,
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
        return Tensor._from_backend(backend, requires_grad=requires_grad)

    @staticmethod
    def full(
        shape: int | tuple[int, ...] | list[int],
        value: int | float | bool,
        *,
        dtype: Dtype | None = None,
        device: Device | str | None = None,
        requires_grad: bool = False,
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
        return Tensor._from_backend(backend, requires_grad=requires_grad)

    @staticmethod
    def arange(
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        *,
        dtype: Dtype | None = None,
        device: Device | str | None = None,
        requires_grad: bool = False,
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
        return Tensor._from_backend(backend, requires_grad=requires_grad)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        dtype: Dtype | None = None,
        device: Device | str | None = None,
        copy: bool = True,
        requires_grad: bool = False,
    ) -> "Tensor":
        """Create a tensor from a NumPy array with explicit copy semantics."""
        if not isinstance(array, np.ndarray):
            raise TensorConstructionError("from_numpy requires a NumPy ndarray input.")
        return Tensor(array, dtype=dtype, device=device, copy=copy, requires_grad=requires_grad)

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
    def requires_grad(self) -> bool:
        """Return whether gradient tracking is enabled for this tensor."""
        return self._requires_grad

    @property
    def grad(self) -> "Tensor | None":
        """Return the gradient accumulated for this tensor, or None if no gradient."""
        return self._grad

    @property
    def grad_fn(self) -> "AutogradNode | None":
        """Return the AutogradNode for this tensor, or None if it's a leaf."""
        return self._grad_fn

    @property
    def is_leaf(self) -> bool:
        """Return whether this tensor is a leaf node in the computation graph."""
        return self._is_leaf

    def backward(self, grad: "Tensor | None" = None) -> None:
        """Compute gradients of this tensor with respect to graph leaves.

        This performs reverse-mode automatic differentiation (backpropagation)
        through the computation graph, accumulating gradients into leaf tensors.

        Args:
            grad: Optional initial gradient. If None and tensor is scalar (shape=()),
                  uses 1.0 as the initial gradient. If None and tensor is non-scalar,
                  raises TensorValidationError.

        Raises:
            TensorValidationError: If tensor is non-scalar and grad is not provided,
                                 or if grad shape is incompatible with tensor shape.
        """
        from titan_ai.tensor.autograd.engine import backward

        backward(self, grad)

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
        return Tensor._from_backend(self._backend.reshape(resolved_shape), requires_grad=self.requires_grad)

    def transpose(self, *axes: int) -> "Tensor":
        """Return a tensor with permuted axes."""
        axis_tuple = tuple(axes) if axes else None
        return Tensor._from_backend(self._backend.transpose_axes(axis_tuple), requires_grad=self.requires_grad)

    def flatten(self) -> "Tensor":
        """Return a 1D copy of the tensor."""
        return Tensor._from_backend(self._backend.flatten(), requires_grad=self.requires_grad)

    def squeeze(self, axis: int | None = None) -> "Tensor":
        """Return a tensor with singleton dimensions removed."""
        return Tensor._from_backend(self._backend.squeeze(axis=axis), requires_grad=self.requires_grad)

    def unsqueeze(self, axis: int) -> "Tensor":
        """Insert a singleton dimension at the given axis."""
        return Tensor._from_backend(self._backend.unsqueeze(axis=axis), requires_grad=self.requires_grad)

    def sum(self, axis: int | None = None, *, keepdims: bool = False) -> "Tensor":
        """Return the sum of elements over the selected axis."""
        from titan_ai.tensor.operations.reductions import reduce_sum

        return reduce_sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: int | None = None, *, keepdims: bool = False) -> "Tensor":
        """Return the mean of elements over the selected axis."""
        from titan_ai.tensor.operations.reductions import reduce_mean

        return reduce_mean(self, axis=axis, keepdims=keepdims)

    def min(self, axis: int | None = None, *, keepdims: bool = False) -> "Tensor":
        """Return the minimum of elements over the selected axis."""
        from titan_ai.tensor.operations.reductions import reduce_min

        return reduce_min(self, axis=axis, keepdims=keepdims)

    def max(self, axis: int | None = None, *, keepdims: bool = False) -> "Tensor":
        """Return the maximum of elements over the selected axis."""
        from titan_ai.tensor.operations.reductions import reduce_max

        return reduce_max(self, axis=axis, keepdims=keepdims)

    def abs(self) -> "Tensor":
        """Return the element-wise absolute value."""
        from titan_ai.tensor.operations.math import abs_values

        return abs_values(self)

    def sqrt(self) -> "Tensor":
        """Return the element-wise square root."""
        from titan_ai.tensor.operations.math import sqrt

        return sqrt(self)

    def exp(self) -> "Tensor":
        """Return the element-wise exponential."""
        from titan_ai.tensor.operations.math import exp

        return exp(self)

    def log(self) -> "Tensor":
        """Return the element-wise natural logarithm."""
        from titan_ai.tensor.operations.math import log

        return log(self)

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
            return Tensor._from_backend(result, requires_grad=self.requires_grad)
        return result

    def __add__(self, other: Any) -> "Tensor":
        from titan_ai.tensor.operations.arithmetic import add

        return add(self, other)

    def __radd__(self, other: Any) -> "Tensor":
        from titan_ai.tensor.operations.arithmetic import add

        return add(self, other)

    def __sub__(self, other: Any) -> "Tensor":
        from titan_ai.tensor.operations.arithmetic import subtract

        return subtract(self, other)

    def __rsub__(self, other: Any) -> "Tensor":
        from titan_ai.tensor.operations.arithmetic import reverse_subtract

        return reverse_subtract(self, other)

    def __mul__(self, other: Any) -> "Tensor":
        from titan_ai.tensor.operations.arithmetic import multiply

        return multiply(self, other)

    def __rmul__(self, other: Any) -> "Tensor":
        from titan_ai.tensor.operations.arithmetic import multiply

        return multiply(self, other)

    def __truediv__(self, other: Any) -> "Tensor":
        from titan_ai.tensor.operations.arithmetic import true_divide

        return true_divide(self, other)

    def __rtruediv__(self, other: Any) -> "Tensor":
        from titan_ai.tensor.operations.arithmetic import reverse_true_divide

        return reverse_true_divide(self, other)

    def __neg__(self) -> "Tensor":
        from titan_ai.tensor.operations.arithmetic import negate

        return negate(self)

    def __abs__(self) -> "Tensor":
        from titan_ai.tensor.operations.math import abs_values

        return abs_values(self)

    def __matmul__(self, other: Any) -> "Tensor":
        from titan_ai.tensor.operations.matmul import matmul

        return matmul(self, other)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype.name}, device={self.device!r})"
