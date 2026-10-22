"""Element-wise mathematical functions for Titan tensors."""

from typing import TYPE_CHECKING, Literal

from titan_ai.tensor.dtypes.dtypes import Dtype
from titan_ai.tensor.operations.arithmetic import require_cpu_device

if TYPE_CHECKING:
    from titan_ai.tensor.core.tensor import Tensor

MathName = Literal["abs", "sqrt", "exp", "log"]


def abs_values(tensor: "Tensor") -> "Tensor":
    """Return the element-wise absolute value."""
    return _unary_math(tensor, "abs")


def sqrt(tensor: "Tensor") -> "Tensor":
    """Return the element-wise square root."""
    return _unary_math(tensor, "sqrt")


def exp(tensor: "Tensor") -> "Tensor":
    """Return the element-wise exponential."""
    return _unary_math(tensor, "exp")


def log(tensor: "Tensor") -> "Tensor":
    """Return the element-wise natural logarithm."""
    return _unary_math(tensor, "log")


def _unary_math(tensor: "Tensor", operation: MathName) -> "Tensor":
    from titan_ai.tensor.core.tensor import Tensor

    require_cpu_device(tensor)
    result_dtype = _math_dtype(tensor.dtype, operation)
    backend_method = getattr(tensor._backend, operation)
    result_backend = backend_method(result_dtype)
    result = Tensor._from_backend(result_backend, requires_grad=tensor.requires_grad)

    # Attach autograd for math operations
    from titan_ai.tensor.autograd.backward import (
        backward_abs,
        backward_exp,
        backward_log,
        backward_sqrt,
    )
    from titan_ai.tensor.autograd.graph import attach_grad_fn, should_build_graph

    if should_build_graph(tensor):
        # Map operation names to backward functions
        backward_map = {
            "abs": backward_abs,
            "sqrt": backward_sqrt,
            "exp": backward_exp,
            "log": backward_log,
        }

        backward_fn = backward_map.get(operation)
        if backward_fn is not None:
            # For sqrt and exp, save the forward output to avoid recomputation
            metadata = {}
            if operation in ("sqrt", "exp"):
                metadata["output"] = result

            result = attach_grad_fn(
                result,
                operation,
                (tensor,),
                backward_fn,
                metadata,
            )

    return result


def _math_dtype(dtype: Dtype, operation: MathName) -> Dtype:
    if operation == "abs":
        return dtype
    if dtype == Dtype.float32:
        return Dtype.float32
    return Dtype.float64
