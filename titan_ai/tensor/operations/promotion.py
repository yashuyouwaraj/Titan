"""Deterministic dtype promotion for Day 4 arithmetic."""

from titan_ai.tensor.dtypes.dtypes import Dtype
from titan_ai.tensor.exceptions.errors import UnsupportedDtypeError, UnsupportedOperandError

_PYTHON_SCALAR_TYPES = (bool, int, float)

# Tensor/Tensor promotion for the five Titan dtypes. Mixed integer/float
# pairs promote to float64 to preserve integer magnitude (NumPy-compatible
# for this dtype subset).
_PROMOTION_TABLE: dict[tuple[Dtype, Dtype], Dtype] = {
    (Dtype.bool, Dtype.bool): Dtype.bool,
    (Dtype.bool, Dtype.int32): Dtype.int32,
    (Dtype.bool, Dtype.int64): Dtype.int64,
    (Dtype.bool, Dtype.float32): Dtype.float32,
    (Dtype.bool, Dtype.float64): Dtype.float64,
    (Dtype.int32, Dtype.int32): Dtype.int32,
    (Dtype.int32, Dtype.int64): Dtype.int64,
    (Dtype.int32, Dtype.float32): Dtype.float64,
    (Dtype.int32, Dtype.float64): Dtype.float64,
    (Dtype.int64, Dtype.int64): Dtype.int64,
    (Dtype.int64, Dtype.float32): Dtype.float64,
    (Dtype.int64, Dtype.float64): Dtype.float64,
    (Dtype.float32, Dtype.float32): Dtype.float32,
    (Dtype.float32, Dtype.float64): Dtype.float64,
    (Dtype.float64, Dtype.float64): Dtype.float64,
}


def promote_dtypes(dtype_a: Dtype, dtype_b: Dtype) -> Dtype:
    """Return the result dtype of a Tensor/Tensor arithmetic operation."""
    key = (dtype_a, dtype_b)
    if key in _PROMOTION_TABLE:
        return _PROMOTION_TABLE[key]
    reversed_key = (dtype_b, dtype_a)
    if reversed_key in _PROMOTION_TABLE:
        return _PROMOTION_TABLE[reversed_key]
    raise UnsupportedDtypeError(
        f"Unsupported dtype promotion between {dtype_a.name} and {dtype_b.name}."
    )


def promote_for_truediv(dtype_a: Dtype, dtype_b: Dtype) -> Dtype:
    """Return the result dtype of true division.

    Integer and boolean results are promoted to ``float64``. Floating-point
    results keep the promoted floating dtype (``float32`` or ``float64``).
    """
    promoted = promote_dtypes(dtype_a, dtype_b)
    if promoted in (Dtype.bool, Dtype.int32, Dtype.int64):
        return Dtype.float64
    return promoted


def is_supported_scalar(value: object) -> bool:
    """Return whether ``value`` is a Titan-supported scalar operand."""
    if type(value) in _PYTHON_SCALAR_TYPES:
        return True
    return _numpy_scalar_dtype(value) is not None


def promote_tensor_and_scalar(tensor_dtype: Dtype, scalar: object) -> Dtype:
    """Return the result dtype of a Tensor/scalar arithmetic operation.

    Python scalars are weak: they do not promote a floating tensor to a
    wider float, and a Python ``int`` does not widen ``int32`` to ``int64``.
    NumPy scalars are strong and use the Tensor/Tensor promotion table.
    """
    if type(scalar) is bool:
        return promote_dtypes(tensor_dtype, Dtype.bool)
    if type(scalar) is int:
        if tensor_dtype == Dtype.bool:
            return Dtype.int64
        return tensor_dtype
    if type(scalar) is float:
        if tensor_dtype in (Dtype.float32, Dtype.float64):
            return tensor_dtype
        return Dtype.float64

    numpy_dtype = _numpy_scalar_dtype(scalar)
    if numpy_dtype is None:
        raise UnsupportedOperandError(f"Unsupported scalar operand {type(scalar).__name__!r}.")
    return promote_dtypes(tensor_dtype, numpy_dtype)


def dtype_of_scalar(scalar: object) -> Dtype:
    """Map a supported scalar to a Titan dtype used when wrapping storage."""
    if type(scalar) is bool:
        return Dtype.bool
    if type(scalar) is int:
        return Dtype.int64
    if type(scalar) is float:
        return Dtype.float64
    numpy_dtype = _numpy_scalar_dtype(scalar)
    if numpy_dtype is None:
        raise UnsupportedOperandError(f"Unsupported scalar operand {type(scalar).__name__!r}.")
    return numpy_dtype


def _numpy_scalar_dtype(value: object) -> Dtype | None:
    """Return a Titan dtype for a NumPy scalar, or ``None`` if not applicable."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    if not isinstance(value, np.generic):
        return None
    try:
        return Dtype.from_numpy(np.dtype(value.dtype))
    except UnsupportedDtypeError:
        return None
