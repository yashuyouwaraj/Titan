"""Backward functions for autograd operations."""

from typing import TYPE_CHECKING

from titan_ai.tensor.operations.broadcast import _align_shapes

if TYPE_CHECKING:
    from titan_ai.tensor.core.tensor import Tensor


def backward_add(
    grad_output: "Tensor",
    x: "Tensor | None",
    y: "Tensor | None",
    metadata: dict | None = None,
) -> tuple["Tensor | None", "Tensor | None"]:
    """Backward pass for addition.

    For z = x + y:
    dz/dx = 1, dz/dy = 1

    Args:
        grad_output: Gradient flowing back from the output.
        x: First input tensor (may be None if garbage collected).
        y: Second input tensor (may be None if garbage collected).
        metadata: Optional metadata (not used for addition).

    Returns:
        Tuple of (grad_x, grad_y). Each is None if the corresponding input
        was None or doesn't require gradients.
    """
    grad_x = None
    grad_y = None

    if x is not None and x.requires_grad:
        grad_x = grad_output

    if y is not None and y.requires_grad:
        grad_y = grad_output

    return grad_x, grad_y


def backward_subtract(
    grad_output: "Tensor",
    x: "Tensor | None",
    y: "Tensor | None",
    metadata: dict | None = None,
) -> tuple["Tensor | None", "Tensor | None"]:
    """Backward pass for subtraction.

    For z = x - y:
    dz/dx = 1, dz/dy = -1

    Args:
        grad_output: Gradient flowing back from the output.
        x: First input tensor (may be None if garbage collected).
        y: Second input tensor (may be None if garbage collected).
        metadata: Optional metadata (not used for subtraction).

    Returns:
        Tuple of (grad_x, grad_y).
    """
    grad_x = None
    grad_y = None

    if x is not None and x.requires_grad:
        grad_x = grad_output

    if y is not None and y.requires_grad:
        grad_y = -grad_output

    return grad_x, grad_y


def backward_multiply(
    grad_output: "Tensor",
    x: "Tensor | None",
    y: "Tensor | None",
    metadata: dict | None = None,
) -> tuple["Tensor | None", "Tensor | None"]:
    """Backward pass for multiplication.

    For z = x * y:
    dz/dx = grad_output * y
    dz/dy = grad_output * x

    Args:
        grad_output: Gradient flowing back from the output.
        x: First input tensor (may be None if garbage collected).
        y: Second input tensor (may be None if garbage collected).
        metadata: Optional metadata with saved tensor values.

    Returns:
        Tuple of (grad_x, grad_y).
    """
    grad_x = None
    grad_y = None

    if x is not None and x.requires_grad:
        if y is not None:
            grad_x = grad_output * y
        elif metadata and "right_value" in metadata:
            # Use saved value from metadata
            from titan_ai.tensor.core.tensor import Tensor
            y_value = metadata["right_value"]
            if y_value is not None:
                y_tensor = Tensor(y_value, dtype=grad_output.dtype, device=grad_output.device, requires_grad=False)
                grad_x = grad_output * y_tensor

    if y is not None and y.requires_grad:
        if x is not None:
            grad_y = grad_output * x
        elif metadata and "left_value" in metadata:
            # Use saved value from metadata
            from titan_ai.tensor.core.tensor import Tensor
            x_value = metadata["left_value"]
            if x_value is not None:
                x_tensor = Tensor(x_value, dtype=grad_output.dtype, device=grad_output.device, requires_grad=False)
                grad_y = grad_output * x_tensor

    return grad_x, grad_y


def backward_true_divide(
    grad_output: "Tensor",
    x: "Tensor | None",
    y: "Tensor | None",
    metadata: dict | None = None,
) -> tuple["Tensor | None", "Tensor | None"]:
    """Backward pass for true division.

    For z = x / y:
    dz/dx = grad_output / y
    dz/dy = -grad_output * x / (y * y)

    Args:
        grad_output: Gradient flowing back from the output.
        x: First input tensor (may be None if garbage collected).
        y: Second input tensor (may be None if garbage collected).
        metadata: Optional metadata with saved tensor values.

    Returns:
        Tuple of (grad_x, grad_y).
    """
    grad_x = None
    grad_y = None

    if x is not None and x.requires_grad:
        if y is not None:
            grad_x = grad_output / y
        elif metadata and "right_value" in metadata:
            from titan_ai.tensor.core.tensor import Tensor
            y_value = metadata["right_value"]
            if y_value is not None:
                y_tensor = Tensor(y_value, dtype=grad_output.dtype, device=grad_output.device, requires_grad=False)
                grad_x = grad_output / y_tensor

    if y is not None and y.requires_grad:
        if x is not None:
            grad_y = -grad_output * x / (y * y)
        elif metadata and "left_value" in metadata:
            from titan_ai.tensor.core.tensor import Tensor
            x_value = metadata["left_value"]
            if x_value is not None:
                x_tensor = Tensor(x_value, dtype=grad_output.dtype, device=grad_output.device, requires_grad=False)
                grad_y = -grad_output * x_tensor / (x_tensor * x_tensor)

    return grad_x, grad_y


def backward_negate(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for negation.

    For z = -x:
    dz/dx = -grad_output

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata (not used for negation).

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        return -grad_output
    return None


def unbroadcast_gradient(grad: "Tensor", original_shape: tuple[int, ...]) -> "Tensor":
    """Reduce a gradient from broadcasted shape back to original shape.

    When an input is broadcasted during forward pass, the gradient must be
    reduced back to the original shape by summing over broadcasted dimensions.

    Args:
        grad: Gradient tensor with the broadcasted output shape.
        original_shape: The original shape of the input before broadcasting.

    Returns:
        Gradient reduced to the original shape.
    """
    if grad.shape == original_shape:
        return grad

    # Handle scalar case
    if original_shape == ():
        return grad.sum()

    # Align shapes to compare dimensions
    aligned_grad_shape, aligned_original_shape = _align_shapes(grad.shape, original_shape)

    # First, sum over broadcasted dimensions (keepdims=True)
    for axis in range(len(aligned_grad_shape)):
        if axis < len(aligned_original_shape):
            if aligned_original_shape[axis] == 1 and aligned_grad_shape[axis] > 1:
                grad = grad.sum(axis=axis, keepdims=True)

    # Then, sum over leading dimensions that were added (keepdims=False)
    # Do this one axis at a time since tuple axes aren't supported
    ndim_diff = len(grad.shape) - len(original_shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0, keepdims=False)

    return grad
