"""Computation graph node for autograd backward pass."""

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from titan_ai.tensor.autograd.context import is_tracking

if TYPE_CHECKING:
    from titan_ai.tensor.core.tensor import Tensor


class AutogradNode:
    """Node in the computation graph that stores backward pass information.
    
    Each AutogradNode represents an operation that produced a tensor.
    It stores references to the input tensors (via weak references to avoid cycles),
    the operation name, and a callable that will compute gradients during backward.
    """

    def __init__(
        self,
        operation: str,
        inputs: tuple["Tensor", ...],
        backward_fn: Callable,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an AutogradNode.
        
        Args:
            operation: Name of the operation (e.g., "add", "multiply").
            inputs: Tuple of input tensors to the operation.
            backward_fn: Callable that computes gradients for this operation.
            metadata: Optional dictionary with additional operation metadata.
        """
        self.operation = operation
        self._inputs = tuple(weakref.ref(t) for t in inputs)
        self._output: "weakref.ref | None" = None  # Set by attach_grad_fn
        self.backward_fn = backward_fn
        self.metadata = metadata or {}

    @property
    def inputs(self) -> tuple["Tensor | None", ...]:
        """Return the input tensors (may be None if already garbage collected)."""
        return tuple(ref() for ref in self._inputs)

    @property
    def output(self) -> "Tensor | None":
        """Return the output tensor (may be None if garbage collected)."""
        if self._output is None:
            return None
        return self._output()

    def __repr__(self) -> str:
        return f"AutogradNode(operation={self.operation!r}, num_inputs={len(self._inputs)})"


def should_build_graph(*inputs: "Tensor") -> bool:
    """Determine if a computation graph should be built for an operation.
    
    A graph is built only when:
    1. Gradient tracking is enabled (default)
    2. At least one input tensor requires gradients
    
    Args:
        *inputs: Input tensors to the operation.
        
    Returns:
        True if graph should be built, False otherwise.
    """
    if not is_tracking():
        return False
    return any(t.requires_grad for t in inputs if t is not None)


def attach_grad_fn(
    output: "Tensor",
    operation: str,
    inputs: tuple["Tensor", ...],
    backward_fn: Callable,
    metadata: dict[str, Any] | None = None,
) -> "Tensor":
    """Attach an AutogradNode to an output tensor.
    
    This marks the output as non-leaf and stores the backward function.
    
    Args:
        output: The output tensor from the operation.
        operation: Name of the operation.
        inputs: Input tensors to the operation.
        backward_fn: Callable for backward pass.
        metadata: Optional operation metadata.
        
    Returns:
        The output tensor with grad_fn attached.
    """
    node = AutogradNode(operation, inputs, backward_fn, metadata)
    node._output = weakref.ref(output)  # Store weak reference to output
    output._grad_fn = node
    output._is_leaf = False
    output._requires_grad = True
    return output
