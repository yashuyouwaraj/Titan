"""Axis normalization for tensor reductions."""

from titan_ai.tensor.exceptions.errors import InvalidAxisError


def normalize_reduction_axis(axis: int | tuple[int, ...] | None, ndim: int) -> int | tuple[int, ...] | None:
    """Validate and canonicalize a reduction axis.

    ``None`` means reduce every dimension. A negative integer is converted
    to the equivalent positive index. A tuple of integers reduces multiple axes.

    Raises:
        InvalidAxisError: if ``axis`` is not ``None``, an integer, or a tuple of
            valid integers for ``ndim``.
    """
    if axis is None:
        return None
    
    # Handle tuple of axes
    if isinstance(axis, tuple):
        if not axis:
            raise InvalidAxisError(f"Reduction axis tuple cannot be empty, got {axis!r}.")
        normalized = []
        for ax in axis:
            if isinstance(ax, bool) or not isinstance(ax, int):
                raise InvalidAxisError(f"Reduction axis must be an integer, got {ax!r} in tuple {axis!r}.")
            if ndim == 0:
                raise InvalidAxisError(f"Axis {ax} is invalid for a 0-dimensional tensor; use axis=None.")
            if ax < -ndim or ax >= ndim:
                raise InvalidAxisError(f"Axis {ax} is out of bounds for tensor of ndim {ndim}.")
            if ax < 0:
                normalized.append(ax + ndim)
            else:
                normalized.append(ax)
        # Remove duplicates and sort
        return tuple(sorted(set(normalized)))
    
    # Handle single axis
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise InvalidAxisError(f"Reduction axis must be None, an integer, or a tuple, got {axis!r}.")
    if ndim == 0:
        raise InvalidAxisError(f"Axis {axis} is invalid for a 0-dimensional tensor; use axis=None.")
    if axis < -ndim or axis >= ndim:
        raise InvalidAxisError(f"Axis {axis} is out of bounds for tensor of ndim {ndim}.")
    if axis < 0:
        return axis + ndim
    return axis


def reduced_element_count(shape: tuple[int, ...], axis: int | tuple[int, ...] | None) -> int:
    """Return how many elements participate in each reduced output value."""
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
