"""Element-wise broadcasting validation for Day 4 arithmetic."""

from titan_ai.tensor.exceptions.errors import BroadcastError


def broadcast_shapes(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]:
    """Return the NumPy-compatible broadcast output shape.

    Dimensions are compared from the trailing axis. Two dimensions are
    compatible when they are equal or when either is ``1``. A scalar
    (shape ``()``) broadcasts with any shape.

    Raises:
        BroadcastError: if the shapes cannot be broadcast together.
    """
    aligned_a, aligned_b = _align_shapes(shape_a, shape_b)
    result: list[int] = []
    for dim_a, dim_b in zip(aligned_a, aligned_b, strict=True):
        if dim_a == dim_b:
            result.append(dim_a)
        elif dim_a == 1:
            result.append(dim_b)
        elif dim_b == 1:
            result.append(dim_a)
        else:
            raise BroadcastError(
                f"Incompatible broadcast shapes {shape_a} and {shape_b}: "
                f"dimension {dim_a} is incompatible with {dim_b}."
            )
    return tuple(result)


def _align_shapes(
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rank = max(len(shape_a), len(shape_b))
    padded_a = (1,) * (rank - len(shape_a)) + shape_a
    padded_b = (1,) * (rank - len(shape_b)) + shape_b
    return padded_a, padded_b
