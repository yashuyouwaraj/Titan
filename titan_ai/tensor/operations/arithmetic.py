"""Element-wise arithmetic dispatch for Titan tensors."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from titan_ai.tensor.backend.base import TensorBackend
from titan_ai.tensor.dtypes.dtypes import Dtype
from titan_ai.tensor.exceptions.errors import (
    DeviceMismatchError,
    UnsupportedDeviceError,
    UnsupportedOperandError,
)
from titan_ai.tensor.operations.broadcast import broadcast_shapes
from titan_ai.tensor.operations.promotion import (
    dtype_of_scalar,
    is_supported_scalar,
    promote_dtypes,
    promote_for_truediv,
    promote_tensor_and_scalar,
)

if TYPE_CHECKING:
    from titan_ai.tensor.core.tensor import Tensor

BackendBinary = Callable[[TensorBackend, Dtype], TensorBackend]


def add(left: "Tensor", right: Any) -> "Tensor":
    """Return element-wise addition of ``left`` and ``right``."""
    left_tensor, right_tensor, result_dtype = _coerce_operands(left, right, "add")
    return _apply_binary(left_tensor, right_tensor, "add", result_dtype)


def subtract(left: "Tensor", right: Any) -> "Tensor":
    """Return element-wise subtraction of ``right`` from ``left``."""
    left_tensor, right_tensor, result_dtype = _coerce_operands(left, right, "subtract")
    return _apply_binary(left_tensor, right_tensor, "subtract", result_dtype)


def multiply(left: "Tensor", right: Any) -> "Tensor":
    """Return element-wise multiplication of ``left`` and ``right``."""
    left_tensor, right_tensor, result_dtype = _coerce_operands(left, right, "multiply")
    return _apply_binary(left_tensor, right_tensor, "multiply", result_dtype)


def true_divide(left: "Tensor", right: Any) -> "Tensor":
    """Return element-wise true division of ``left`` by ``right``."""
    left_tensor, right_tensor, result_dtype = _coerce_operands(left, right, "true_divide")
    return _apply_binary(left_tensor, right_tensor, "true_divide", result_dtype)


def reverse_subtract(tensor: "Tensor", left_operand: Any) -> "Tensor":
    """Return ``left_operand - tensor``."""
    left_tensor, right_tensor, result_dtype = _coerce_left_operand(left_operand, tensor, "subtract")
    return _apply_binary(left_tensor, right_tensor, "subtract", result_dtype)


def reverse_true_divide(tensor: "Tensor", left_operand: Any) -> "Tensor":
    """Return ``left_operand / tensor``."""
    left_tensor, right_tensor, result_dtype = _coerce_left_operand(
        left_operand, tensor, "true_divide"
    )
    return _apply_binary(left_tensor, right_tensor, "true_divide", result_dtype)


def negate(tensor: "Tensor") -> "Tensor":
    """Return element-wise negation of ``tensor``."""
    from titan_ai.tensor.core.tensor import Tensor

    require_cpu_device(tensor)
    result_dtype = Dtype.int64 if tensor.dtype == Dtype.bool else tensor.dtype
    result_backend = tensor._backend.negate(result_dtype)
    return Tensor._from_backend(result_backend)


def _apply_binary(
    left: "Tensor",
    right: "Tensor",
    operation: str,
    result_dtype: Dtype,
) -> "Tensor":
    from titan_ai.tensor.core.tensor import Tensor

    require_same_cpu_device(left, right)
    broadcast_shapes(left.shape, right.shape)
    backend_op: BackendBinary = getattr(left._backend, operation)
    result_backend = backend_op(right._backend, result_dtype)
    return Tensor._from_backend(result_backend)


def _coerce_operands(
    left: "Tensor",
    right: Any,
    operation: str,
) -> tuple["Tensor", "Tensor", Dtype]:
    from titan_ai.tensor.core.tensor import Tensor

    if isinstance(right, Tensor):
        return left, right, _result_dtype(left.dtype, right.dtype, operation)
    if is_supported_scalar(right):
        result_dtype = _result_dtype_with_scalar(left.dtype, right, operation)
        return left, _scalar_tensor(right, left), result_dtype
    raise UnsupportedOperandError(
        f"Unsupported operand type {type(right).__name__!r} for tensor {operation}."
    )


def _coerce_left_operand(
    left_operand: Any,
    right: "Tensor",
    operation: str,
) -> tuple["Tensor", "Tensor", Dtype]:
    from titan_ai.tensor.core.tensor import Tensor

    if isinstance(left_operand, Tensor):
        return left_operand, right, _result_dtype(left_operand.dtype, right.dtype, operation)
    if is_supported_scalar(left_operand):
        result_dtype = _result_dtype_with_scalar(right.dtype, left_operand, operation)
        return _scalar_tensor(left_operand, right), right, result_dtype
    raise UnsupportedOperandError(
        f"Unsupported operand type {type(left_operand).__name__!r} for tensor {operation}."
    )


def _scalar_tensor(scalar: object, reference: "Tensor") -> "Tensor":
    from titan_ai.tensor.core.tensor import Tensor

    return Tensor(scalar, dtype=dtype_of_scalar(scalar), device=reference.device)


def _result_dtype(left: Dtype, right: Dtype, operation: str) -> Dtype:
    if operation == "true_divide":
        return promote_for_truediv(left, right)
    return promote_dtypes(left, right)


def _result_dtype_with_scalar(tensor_dtype: Dtype, scalar: object, operation: str) -> Dtype:
    arithmetic_dtype = promote_tensor_and_scalar(tensor_dtype, scalar)
    if operation == "true_divide":
        return promote_for_truediv(arithmetic_dtype, arithmetic_dtype)
    return arithmetic_dtype


def require_same_cpu_device(left: "Tensor", right: "Tensor") -> None:
    """Require both tensors to share a CPU device; never auto-move data."""
    if left.device != right.device:
        raise DeviceMismatchError(
            f"Tensors must be on the same device, got {left.device!r} and {right.device!r}."
        )
    require_cpu_device(left)
    require_cpu_device(right)


def require_cpu_device(tensor: "Tensor") -> None:
    """Require CPU execution for Day 4 operations."""
    if tensor.device.type != "cpu":
        raise UnsupportedDeviceError(
            f"Unsupported device {tensor.device!r} for this operation. "
            "Only CPU execution is available in Week 1. "
            "Tensors are never moved between devices automatically."
        )
