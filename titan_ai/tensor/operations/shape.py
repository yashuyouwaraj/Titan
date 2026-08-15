"""Shape normalization helpers for tensor operations."""

from titan_ai.tensor.exceptions.errors import InvalidShapeError


def normalize_shape(shape: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Normalize a shape argument to a tuple of integers."""
    if isinstance(shape, int):
        if shape < 0:
            raise InvalidShapeError(f"Shape dimensions must be non-negative, got {shape}.")
        return (shape,)
    if isinstance(shape, (tuple, list)):
        if len(shape) == 0:
            return ()
        normalized: list[int] = []
        for dim in shape:
            if not isinstance(dim, int):
                raise InvalidShapeError(f"Shape dimensions must be integers, got {dim!r}.")
            if dim < 0 and dim != -1:
                raise InvalidShapeError(
                    f"Shape dimensions must be non-negative or -1 for inference, got {dim}."
                )
            normalized.append(dim)
        return tuple(normalized)
    raise InvalidShapeError(f"Invalid shape argument: {shape!r}.")


def resolve_inferred_shape(
    current_size: int,
    target_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Resolve a single ``-1`` inference dimension in a target shape."""
    inferred_count = sum(1 for dim in target_shape if dim == -1)
    if inferred_count > 1:
        raise InvalidShapeError(
            f"Only one dimension may be inferred with -1, got shape {target_shape}."
        )

    known_product = 1
    inferred_index: int | None = None
    for index, dim in enumerate(target_shape):
        if dim == -1:
            inferred_index = index
        else:
            known_product *= dim

    if inferred_index is None:
        if known_product != current_size:
            raise InvalidShapeError(
                f"Cannot reshape tensor with {current_size} elements to shape {target_shape}."
            )
        return target_shape

    if known_product == 0 and current_size != 0:
        raise InvalidShapeError(
            f"Cannot infer dimension for shape {target_shape} with {current_size} elements."
        )
    if current_size % known_product != 0:
        raise InvalidShapeError(
            f"Cannot reshape tensor with {current_size} elements to shape {target_shape}."
        )

    inferred_dim = current_size // known_product
    resolved = list(target_shape)
    resolved[inferred_index] = inferred_dim
    return tuple(resolved)
