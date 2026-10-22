"""Numerical gradient checking utility for autograd validation."""

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from titan_ai.tensor.core.tensor import Tensor


def numerical_gradient(
    func: Callable[["Tensor"], "Tensor"],
    x: "Tensor",
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute numerical gradient using central differences.

    For a scalar function f(x), computes:
    df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)

    Args:
        func: Function that takes a Tensor and returns a scalar Tensor.
        x: Input tensor at which to compute gradient.
        eps: Small epsilon for finite differences.

    Returns:
        NumPy array containing the numerical gradient.
    """
    from titan_ai.tensor.autograd.context import disable_tracking, enable_tracking

    x_np = x.numpy()
    grad_np = np.zeros_like(x_np)

    # Disable tracking during numerical gradient computation
    was_tracking = disable_tracking()

    # Temporarily disable requires_grad on x
    original_requires_grad = x._requires_grad
    x._requires_grad = False

    try:
        # Iterate over each element
        for idx in np.ndindex(x_np.shape):
            # Create perturbations
            x_plus = x_np.copy()
            x_minus = x_np.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps

            # Evaluate function at perturbed points
            from titan_ai.tensor.core.tensor import Tensor

            f_plus = func(Tensor(x_plus, dtype=x.dtype, device=x.device, requires_grad=False))
            f_minus = func(Tensor(x_minus, dtype=x.dtype, device=x.device, requires_grad=False))

            # Central difference
            grad_np[idx] = (f_plus.numpy() - f_minus.numpy()) / (2 * eps)
    finally:
        # Restore tracking state and requires_grad
        if was_tracking:
            enable_tracking()
        x._requires_grad = original_requires_grad

    return grad_np


def check_gradient(
    func: Callable[["Tensor"], "Tensor"],
    x: "Tensor",
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    eps: float = 1e-6,
) -> bool:
    """Check analytical gradient against numerical gradient.

    Args:
        func: Function that takes a Tensor and returns a scalar Tensor.
        x: Input tensor with requires_grad=True.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        eps: Small epsilon for finite differences.

    Returns:
        True if gradients match within tolerance, False otherwise.
    """
    # Compute analytical gradient
    y = func(x)
    y.backward()
    analytical_grad = x.grad
    if analytical_grad is None:
        raise ValueError("No analytical gradient computed - set requires_grad=True on input.")

    # Store analytical gradient
    analytical_grad_np = analytical_grad.numpy().copy()

    # Clear gradient for numerical check
    x._grad = None

    # Compute numerical gradient
    numerical_grad_np = numerical_gradient(func, x, eps=eps)

    # Compare
    return np.allclose(
        analytical_grad_np,
        numerical_grad_np,
        atol=atol,
        rtol=rtol,
    )


def max_gradient_error(
    func: Callable[["Tensor"], "Tensor"],
    x: "Tensor",
    *,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Compute maximum absolute and relative error between gradients.

    Args:
        func: Function that takes a Tensor and returns a scalar Tensor.
        x: Input tensor with requires_grad=True.
        eps: Small epsilon for finite differences.

    Returns:
        Tuple of (max_absolute_error, max_relative_error).
    """
    # Compute analytical gradient
    y = func(x)
    y.backward()
    analytical_grad = x.grad
    if analytical_grad is None:
        raise ValueError("No analytical gradient computed - set requires_grad=True on input.")

    # Store analytical gradient
    analytical_grad_np = analytical_grad.numpy().copy()

    # Clear gradient for numerical check
    x._grad = None

    # Compute numerical gradient
    numerical_grad_np = numerical_gradient(func, x, eps=eps)

    # Compute errors
    abs_error = np.abs(analytical_grad_np - numerical_grad_np)
    rel_error = abs_error / (np.abs(numerical_grad_np) + 1e-10)

    return float(np.max(abs_error)), float(np.max(rel_error))
