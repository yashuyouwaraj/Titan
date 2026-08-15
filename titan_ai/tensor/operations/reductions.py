"""Reduction dispatch for Titan tensors."""

from typing import TYPE_CHECKING, Literal

from titan_ai.tensor.dtypes.dtypes import Dtype
from titan_ai.tensor.exceptions.errors import TensorValidationError
from titan_ai.tensor.operations.arithmetic import require_cpu_device
from titan_ai.tensor.operations.axis import normalize_reduction_axis, reduced_element_count

if TYPE_CHECKING:
    from titan_ai.tensor.core.tensor import Tensor

ReductionName = Literal["sum", "mean", "min", "max"]


def reduce_sum(
    tensor: "Tensor",
    axis: int | None = None,
    *,
    keepdims: bool = False,
) -> "Tensor":
    """Return the sum of tensor elements over the selected axis."""
    return _reduce(tensor, "sum", axis, keepdims)


def reduce_mean(
    tensor: "Tensor",
    axis: int | None = None,
    *,
    keepdims: bool = False,
) -> "Tensor":
    """Return the mean of tensor elements over the selected axis."""
    return _reduce(tensor, "mean", axis, keepdims)


def reduce_min(
    tensor: "Tensor",
    axis: int | None = None,
    *,
    keepdims: bool = False,
) -> "Tensor":
    """Return the minimum of tensor elements over the selected axis."""
    return _reduce(tensor, "min", axis, keepdims)


def reduce_max(
    tensor: "Tensor",
    axis: int | None = None,
    *,
    keepdims: bool = False,
) -> "Tensor":
    """Return the maximum of tensor elements over the selected axis."""
    return _reduce(tensor, "max", axis, keepdims)


def _reduce(
    tensor: "Tensor",
    operation: ReductionName,
    axis: int | None,
    keepdims: bool,
) -> "Tensor":
    from titan_ai.tensor.core.tensor import Tensor

    require_cpu_device(tensor)
    normalized_axis = normalize_reduction_axis(axis, tensor.ndim)
    _validate_empty_reduction(tensor.shape, normalized_axis, operation)
    result_dtype = _reduction_dtype(tensor.dtype, operation)
    backend_method = getattr(tensor._backend, f"reduce_{operation}")
    result_backend = backend_method(normalized_axis, keepdims, result_dtype)
    return Tensor._from_backend(result_backend)


def _validate_empty_reduction(
    shape: tuple[int, ...],
    axis: int | None,
    operation: ReductionName,
) -> None:
    if reduced_element_count(shape, axis) != 0:
        return
    if operation == "sum":
        return
    raise TensorValidationError(
        f"Cannot compute {operation} over an empty reduction (shape {shape}, axis={axis})."
    )


def _reduction_dtype(dtype: Dtype, operation: ReductionName) -> Dtype:
    if operation == "sum":
        if dtype in (Dtype.bool, Dtype.int32, Dtype.int64):
            return Dtype.int64
        return dtype
    if operation == "mean":
        if dtype == Dtype.float32:
            return Dtype.float32
        return Dtype.float64
    return dtype
