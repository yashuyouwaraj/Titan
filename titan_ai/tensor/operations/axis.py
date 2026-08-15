"""Axis normalization for tensor reductions."""

from titan_ai.tensor.exceptions.errors import InvalidAxisError


def normalize_reduction_axis(axis: int | None, ndim: int) -> int | None:
    """Validate and canonicalize a reduction axis.

    ``None`` means reduce every dimension. A negative integer is converted
    to the equivalent positive index. Tuple axes are not supported in Day 5.

    Raises:
        InvalidAxisError: if ``axis`` is not ``None`` or a valid integer for
            ``ndim``.
    """
    if axis is None:
        return None
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise InvalidAxisError(f"Reduction axis must be None or an integer, got {axis!r}.")
    if ndim == 0:
        raise InvalidAxisError(f"Axis {axis} is invalid for a 0-dimensional tensor; use axis=None.")
    if axis < -ndim or axis >= ndim:
        raise InvalidAxisError(f"Axis {axis} is out of bounds for tensor of ndim {ndim}.")
    if axis < 0:
        return axis + ndim
    return axis


def reduced_element_count(shape: tuple[int, ...], axis: int | None) -> int:
    """Return how many elements participate in each reduced output value."""
    if axis is None:
        if not shape:
            return 1
        total = 1
        for dim in shape:
            total *= dim
        return total
    return shape[axis]
