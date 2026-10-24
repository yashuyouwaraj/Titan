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
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
) -> "Tensor":
    """Return the sum of tensor elements over the selected axis."""
    return _reduce(tensor, "sum", axis, keepdims)


def reduce_mean(
    tensor: "Tensor",
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
) -> "Tensor":
    """Return the mean of tensor elements over the selected axis."""
    return _reduce(tensor, "mean", axis, keepdims)


def reduce_min(
    tensor: "Tensor",
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
) -> "Tensor":
    """Return the minimum of tensor elements over the selected axis."""
    return _reduce(tensor, "min", axis, keepdims)


def reduce_max(
    tensor: "Tensor",
    axis: int | tuple[int, ...] | None = None,
    *,
    keepdims: bool = False,
) -> "Tensor":
    """Return the maximum of tensor elements over the selected axis."""
    return _reduce(tensor, "max", axis, keepdims)


def _reduce(
    tensor: "Tensor",
    operation: ReductionName,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> "Tensor":
    from titan_ai.tensor.core.tensor import Tensor

    require_cpu_device(tensor)
    normalized_axis = normalize_reduction_axis(axis, tensor.ndim)
    _validate_empty_reduction(tensor.shape, normalized_axis, operation)
    result_dtype = _reduction_dtype(tensor.dtype, operation)
    backend_method = getattr(tensor._backend, f"reduce_{operation}")
    result_backend = backend_method(normalized_axis, keepdims, result_dtype)
    result = Tensor._from_backend(result_backend, requires_grad=tensor.requires_grad)

    # Attach autograd for sum, mean, min, and max
    if operation in ("sum", "mean", "min", "max") and tensor.requires_grad:
        from titan_ai.tensor.autograd.backward import (
            backward_sum,
            backward_mean,
            backward_min,
            backward_max,
        )
        from titan_ai.tensor.autograd.graph import attach_grad_fn, should_build_graph

        if should_build_graph(tensor):
            # Calculate metadata for backward
            original_shape = tensor.shape
            reduced_axes = _get_reduced_axes(tensor.ndim, normalized_axis)

            if operation in ("sum", "mean"):
                reduction_size = _get_reduction_size(tensor.shape, normalized_axis)
                metadata = {
                    "original_shape": original_shape,
                    "reduced_axes": reduced_axes,
                    "reduction_size": reduction_size,
                }
                backward_fn = backward_sum if operation == "sum" else backward_mean
            else:  # min or max
                metadata = {
                    "original_shape": original_shape,
                    "reduced_axes": reduced_axes,
                    "keepdims": keepdims,
                }
                backward_fn = backward_min if operation == "min" else backward_max

            result = attach_grad_fn(
                result,
                operation,
                (tensor,),
                backward_fn,
                metadata,
            )

    return result


def _get_reduced_axes(ndim: int, axis: int | tuple[int, ...] | None) -> list[int]:
    """Get the list of axes that were reduced."""
    if axis is None:
        return list(range(ndim))
    if isinstance(axis, tuple):
        return list(axis)
    return [axis]


def _get_reduction_size(shape: tuple[int, ...], axis: int | tuple[int, ...] | None) -> int:
    """Get the number of elements in the reduction."""
    if axis is None:
        if not shape:
            return 1
        total = 1
        for dim in shape:
            total *= dim
        return total
    if isinstance(axis, tuple):
        total = 1
        for ax in axis:
            total *= shape[ax]
        return total
    return shape[axis]


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
