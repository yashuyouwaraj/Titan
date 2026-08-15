"""Matrix multiplication dispatch for Titan tensors."""

from typing import TYPE_CHECKING

from titan_ai.tensor.dtypes.dtypes import Dtype
from titan_ai.tensor.exceptions.errors import InvalidShapeError, UnsupportedOperandError
from titan_ai.tensor.operations.arithmetic import require_same_cpu_device
from titan_ai.tensor.operations.promotion import promote_dtypes

if TYPE_CHECKING:
    from titan_ai.tensor.core.tensor import Tensor

_MAX_MATMUL_NDIM = 2


def matmul(left: "Tensor", right: object) -> "Tensor":
    """Return the matrix product of ``left`` and ``right``.

    Supported ranks: 1D and 2D (including mixed 1D/2D). Higher-rank
    batched matmul is not implemented in Day 4.
    """
    from titan_ai.tensor.core.tensor import Tensor

    if not isinstance(right, Tensor):
        raise UnsupportedOperandError(
            f"Matrix multiplication requires Tensor operands, got {type(right).__name__}."
        )
    require_same_cpu_device(left, right)
    matmul_result_shape(left.shape, right.shape)
    result_dtype = _matmul_dtype(left, right)
    result_backend = left._backend.matmul(right._backend, result_dtype)
    return Tensor._from_backend(result_backend)


def matmul_result_shape(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]:
    """Validate matmul ranks and inner dimensions; return the output shape."""
    ndim_a = len(shape_a)
    ndim_b = len(shape_b)
    if ndim_a == 0 or ndim_b == 0:
        raise InvalidShapeError(
            "Matrix multiplication requires at least 1D tensors, "
            f"got shapes {shape_a} and {shape_b}."
        )
    if ndim_a > _MAX_MATMUL_NDIM or ndim_b > _MAX_MATMUL_NDIM:
        raise InvalidShapeError(
            "Matrix multiplication currently supports 1D and 2D tensors only, "
            f"got shapes {shape_a} and {shape_b}."
        )

    if ndim_a == 1 and ndim_b == 1:
        _require_inner_dims(shape_a[0], shape_b[0], shape_a, shape_b)
        return ()
    if ndim_a == 2 and ndim_b == 2:
        _require_inner_dims(shape_a[1], shape_b[0], shape_a, shape_b)
        return (shape_a[0], shape_b[1])
    if ndim_a == 1 and ndim_b == 2:
        _require_inner_dims(shape_a[0], shape_b[0], shape_a, shape_b)
        return (shape_b[1],)
    _require_inner_dims(shape_a[1], shape_b[0], shape_a, shape_b)
    return (shape_a[0],)


def _require_inner_dims(
    left_inner: int,
    right_inner: int,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
) -> None:
    if left_inner != right_inner:
        raise InvalidShapeError(
            f"Incompatible matrix multiplication shapes {shape_a} and {shape_b}: "
            f"inner dimensions {left_inner} and {right_inner} must match."
        )


def _matmul_dtype(left: "Tensor", right: "Tensor") -> Dtype:
    promoted = promote_dtypes(left.dtype, right.dtype)
    if promoted == Dtype.bool:
        return Dtype.int64
    return promoted
