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


def backward_matmul(
    grad_output: "Tensor",
    left: "Tensor | None",
    right: "Tensor | None",
    metadata: dict | None = None,
) -> tuple["Tensor | None", "Tensor | None"]:
    """Backward pass for matrix multiplication.

    For C = A @ B:
    - 1D @ 1D → scalar: dA = grad_output * B, dB = grad_output * A
    - 2D @ 2D → 2D: dA = grad_output @ B.T, dB = A.T @ grad_output
    - 1D @ 2D → 1D: dA = grad_output @ B.T (then reshape to 1D), dB = A.T @ grad_output
    - 2D @ 1D → 1D: dA = grad_output @ B.T (reshape B to 2D), dB = A.T @ grad_output

    Args:
        grad_output: Gradient flowing back from the output.
        left: First input tensor (may be None if garbage collected).
        right: Second input tensor (may be None if garbage collected).
        metadata: Optional metadata with input shapes.

    Returns:
        Tuple of (grad_left, grad_right).
    """
    grad_left = None
    grad_right = None

    if left is not None and left.requires_grad and right is not None:
        left_ndim = left.ndim
        right_ndim = right.ndim

        if left_ndim == 1 and right_ndim == 1:
            # 1D @ 1D → scalar
            # dA = grad_output * B
            grad_left = grad_output * right
        elif left_ndim == 2 and right_ndim == 2:
            # 2D @ 2D → 2D
            # dA = grad_output @ B.T
            grad_left = grad_output @ right.T
        elif left_ndim == 1 and right_ndim == 2:
            # 1D @ 2D → 1D
            # dA = grad_output @ B.T, then reshape to match original left shape
            # grad_output is (M,), B.T is (N, M), need to unsqueeze grad_output to (1, M)
            grad_output_2d = grad_output.unsqueeze(0)
            # Result is (1, N), squeeze to (N,)
            grad_left = (grad_output_2d @ right.T).squeeze(0)
        elif left_ndim == 2 and right_ndim == 1:
            # 2D @ 1D → 1D
            # dA = grad_output @ B.T (B is 1D, need to treat as column vector)
            # grad_output is (M,), B is (N,)
            # Need: grad_output @ B.T = (M,) @ (N,) doesn't work
            # Correct: unsqueeze grad_output to (M, 1), B to (1, N), then matmul
            grad_output_2d = grad_output.unsqueeze(1)
            right_2d = right.unsqueeze(0)
            # Result is (M, N), matches left shape
            grad_left = grad_output_2d @ right_2d

    if right is not None and right.requires_grad and left is not None:
        left_ndim = left.ndim
        right_ndim = right.ndim

        if left_ndim == 1 and right_ndim == 1:
            # 1D @ 1D → scalar
            # dB = grad_output * A
            grad_right = grad_output * left
        elif left_ndim == 2 and right_ndim == 2:
            # 2D @ 2D → 2D
            # dB = A.T @ grad_output
            grad_right = left.T @ grad_output
        elif left_ndim == 1 and right_ndim == 2:
            # 1D @ 2D → 1D
            # dB = A.T @ grad_output (A is 1D, need to treat as row vector)
            # A is (N,), grad_output is (M,)
            # Need outer product: A[i] * grad_output[j] for each i, j
            # This is A[:, None] @ grad_output[None, :] = (N, 1) @ (1, M) = (N, M)
            left_2d = left.unsqueeze(1)  # (N, 1)
            grad_output_2d = grad_output.unsqueeze(0)  # (1, M)
            # Result is (N, M), matches right shape
            grad_right = left_2d @ grad_output_2d
        elif left_ndim == 2 and right_ndim == 1:
            # 2D @ 1D → 1D
            # dB = A.T @ grad_output
            # A.T is (N, M), grad_output is (M,), result is (N,) - matches right shape
            grad_right = left.T @ grad_output

    return grad_left, grad_right


def backward_sum(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for sum reduction.

    For y = sum(x):
    The gradient is distributed by broadcasting grad_output to match x's shape.

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata with original shape and reduction parameters.

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        original_shape = metadata.get("original_shape") if metadata else None
        if original_shape is None:
            original_shape = x.shape

        # Broadcast grad_output to original shape
        # If keepdims was True, grad_output already has the right number of dimensions
        # If keepdims was False, we need to unsqueeze
        grad = grad_output
        if grad.ndim < len(original_shape):
            # Need to unsqueeze dimensions that were reduced
            reduced_axes = metadata.get("reduced_axes") if metadata else None
            if reduced_axes is not None:
                # unsqueeze each reduced axis
                for axis in sorted(reduced_axes):
                    grad = grad.unsqueeze(axis)

        # Now broadcast to full shape
        if grad.shape != original_shape:
            # If grad is scalar (shape ()) and original is not, we need to broadcast
            if grad.shape == () and original_shape != ():
                # Scalar gradient: broadcast to full shape
                from titan_ai.tensor.core.tensor import Tensor
                import numpy as np
                grad_np = np.full(original_shape, grad.numpy())
                grad = Tensor(grad_np, dtype=grad.dtype, device=grad.device, requires_grad=False)
            else:
                # Need to broadcast grad to original shape
                # Use numpy broadcasting to handle this
                from titan_ai.tensor.core.tensor import Tensor
                import numpy as np
                grad_np = np.broadcast_to(grad.numpy(), original_shape)
                grad = Tensor(grad_np, dtype=grad.dtype, device=grad.device, requires_grad=False)

        return grad
    return None


def backward_mean(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for mean reduction.

    For y = mean(x):
    The gradient is distributed by broadcasting grad_output to match x's shape,
    then divided by the number of elements in the reduction.

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata with original shape and reduction parameters.

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        original_shape = metadata.get("original_shape") if metadata else None
        if original_shape is None:
            original_shape = x.shape

        # Get the number of elements in the reduction
        reduction_size = metadata.get("reduction_size") if metadata else 1
        if reduction_size == 0:
            reduction_size = 1

        # Broadcast grad_output to original shape
        grad = grad_output
        if grad.ndim < len(original_shape):
            # Need to unsqueeze dimensions that were reduced
            reduced_axes = metadata.get("reduced_axes") if metadata else None
            if reduced_axes is not None:
                # unsqueeze each reduced axis
                for axis in sorted(reduced_axes):
                    grad = grad.unsqueeze(axis)

        # Now broadcast to full shape
        if grad.shape != original_shape:
            # If grad is scalar (shape ()) and original is not, we need to broadcast
            if grad.shape == () and original_shape != ():
                # Scalar gradient: broadcast to full shape
                from titan_ai.tensor.core.tensor import Tensor
                import numpy as np
                grad_np = np.full(original_shape, grad.numpy())
                grad = Tensor(grad_np, dtype=grad.dtype, device=grad.device, requires_grad=False)
            else:
                # Need to broadcast grad to original shape
                # Use numpy broadcasting to handle this
                from titan_ai.tensor.core.tensor import Tensor
                import numpy as np
                grad_np = np.broadcast_to(grad.numpy(), original_shape)
                grad = Tensor(grad_np, dtype=grad.dtype, device=grad.device, requires_grad=False)

        # Divide by the number of elements in the reduction
        grad = grad / reduction_size

        return grad
    return None


def backward_abs(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for absolute value.

    For y = abs(x):
    dy/dx = sign(x) for x != 0
    At x = 0, use subgradient of 0 (arbitrary but consistent choice).

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata (not used for abs).

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        # sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
        # Use numpy's sign function
        import numpy as np
        from titan_ai.tensor.core.tensor import Tensor

        x_np = x.numpy()
        sign_np = np.sign(x_np)
        sign = Tensor(sign_np, dtype=grad_output.dtype, device=grad_output.device, requires_grad=False)

        return grad_output * sign
    return None


def backward_sqrt(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for square root.

    For y = sqrt(x):
    dy/dx = 1 / (2 * sqrt(x))

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata with saved forward output.

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        # dy/dx = 1 / (2 * sqrt(x))
        # Prefer using saved forward output if available
        if metadata and "output" in metadata:
            output = metadata["output"]
            if output is not None:
                # grad_output / (2 * output)
                return grad_output / (2 * output)
        else:
            # Recompute sqrt(x)
            from titan_ai.tensor.core.tensor import Tensor
            from titan_ai.tensor.operations.math import sqrt

            sqrt_x = sqrt(x)
            return grad_output / (2 * sqrt_x)
    return None


def backward_exp(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for exponential.

    For y = exp(x):
    dy/dx = exp(x)

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata with saved forward output.

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        # dy/dx = exp(x)
        # Prefer using saved forward output if available
        if metadata and "output" in metadata:
            output = metadata["output"]
            if output is not None:
                # grad_output * exp(x)
                return grad_output * output
        else:
            # Recompute exp(x)
            from titan_ai.tensor.core.tensor import Tensor
            from titan_ai.tensor.operations.math import exp

            exp_x = exp(x)
            return grad_output * exp_x
    return None


def backward_log(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for natural logarithm.

    For y = log(x):
    dy/dx = 1 / x

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata (not used for log).

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        # dy/dx = 1 / x
        return grad_output / x
    return None


def backward_reshape(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for reshape.

    For y = x.reshape(new_shape):
    The gradient is reshaped back to the original input shape.

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata with original shape.

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        original_shape = metadata.get("original_shape") if metadata else None
        if original_shape is None:
            original_shape = x.shape

        # Reshape gradient back to original shape
        return grad_output.reshape(original_shape)
    return None


def backward_transpose(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for transpose.

    For y = x.transpose(axes):
    The gradient is transposed with the inverse permutation.

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata with the axes permutation.

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        axes = metadata.get("axes") if metadata else None

        if axes is None:
            # Default transpose: reverse all axes
            # The inverse of reverse is also reverse
            return grad_output.transpose()
        else:
            # Explicit axes permutation
            # The inverse permutation: axes_inv[axes[i]] = i
            axes_inv = [0] * len(axes)
            for i, ax in enumerate(axes):
                axes_inv[ax] = i
            return grad_output.transpose(*axes_inv)
    return None


def backward_squeeze(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for squeeze.

    For y = x.squeeze(axis):
    The gradient is unsqueezed at the removed axis.

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata with the squeezed axis.

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        axis = metadata.get("axis") if metadata else None

        if axis is None:
            # All singleton dimensions were removed
            # Simply reshape to original shape
            original_shape = metadata.get("original_shape") if metadata else x.shape
            return grad_output.reshape(original_shape)
        else:
            # Specific axis was squeezed
            return grad_output.unsqueeze(axis)
    return None


def backward_unsqueeze(
    grad_output: "Tensor",
    x: "Tensor | None",
    metadata: dict | None = None,
) -> "Tensor | None":
    """Backward pass for unsqueeze.

    For y = x.unsqueeze(axis):
    The gradient is squeezed at the inserted axis.

    Args:
        grad_output: Gradient flowing back from the output.
        x: Input tensor (may be None if garbage collected).
        metadata: Optional metadata with the unsqueezed axis.

    Returns:
        Gradient for x, or None if x was None or doesn't require gradients.
    """
    if x is not None and x.requires_grad:
        axis = metadata.get("axis") if metadata else None
        if axis is not None:
            return grad_output.squeeze(axis=axis)
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
