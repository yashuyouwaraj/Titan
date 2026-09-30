"""Backward engine for autograd reverse-mode differentiation."""

from typing import TYPE_CHECKING

from titan_ai.tensor.autograd.backward import unbroadcast_gradient
from titan_ai.tensor.autograd.context import disable_tracking, enable_tracking, is_tracking
from titan_ai.tensor.exceptions.errors import TensorValidationError

if TYPE_CHECKING:
    from titan_ai.tensor.core.tensor import Tensor


def backward(tensor: "Tensor", grad: "Tensor | None" = None) -> None:
    """Perform backward pass to compute gradients.
    
    This function traverses the computation graph in reverse topological order,
    calling each node's backward function and accumulating gradients into the
    input tensors.
    
    Args:
        tensor: The output tensor from which to start backpropagation.
        grad: Optional initial gradient. If None and tensor is scalar, uses 1.0.
              If None and tensor is non-scalar, raises TensorValidationError.
    
    Raises:
        TensorValidationError: If tensor is non-scalar and grad is not provided,
                             or if grad shape is incompatible with tensor shape.
    """
    if tensor.grad_fn is None:
        # Leaf tensor or no graph - nothing to do
        return
    
    # Validate and set initial gradient
    initial_grad = _validate_initial_gradient(tensor, grad)
    
    # Disable gradient tracking during backward pass
    was_tracking = is_tracking()
    disable_tracking()
    
    try:
        # Build topological order and traverse in reverse
        order = _build_topological_order(tensor)
        
        # Store gradients for each tensor in the graph
        tensor_grads = {id(tensor): initial_grad}
        
        # Process nodes in reverse topological order
        for node in reversed(order):
            # Get the output tensor for this node
            output_tensor = node.output
            if output_tensor is None:
                continue
            
            grad_output = tensor_grads.get(id(output_tensor))
            if grad_output is None:
                continue
            
            # Call the backward function
            input_grads = node.backward_fn(grad_output, *node.inputs, metadata=node.metadata)
            
            # Handle unary operations (single return value)
            if len(node.inputs) == 1:
                input_grads = (input_grads,)
            
            # Accumulate gradients into input tensors
            for input_tensor, input_grad in zip(node.inputs, input_grads):
                if input_tensor is not None and input_grad is not None:
                    # Unbroadcast if needed
                    if input_grad.shape != input_tensor.shape:
                        input_grad = unbroadcast_gradient(input_grad, input_tensor.shape)

                    # Ensure gradient doesn't require grad
                    if input_grad.requires_grad:
                        from titan_ai.tensor.core.tensor import Tensor
                        input_grad = Tensor._from_backend(input_grad._backend, requires_grad=False)

                    # Accumulate gradient
                    if input_tensor._grad is None:
                        input_tensor._grad = input_grad
                    else:
                        # Use backend directly to avoid building new graph
                        from titan_ai.tensor.core.tensor import Tensor
                        new_grad_backend = input_tensor._grad._backend.add(input_grad._backend, input_tensor._grad.dtype)
                        input_tensor._grad = Tensor._from_backend(new_grad_backend, requires_grad=False)

                    # If this input has a grad_fn, record its gradient for propagation
                    if input_tensor.grad_fn is not None:
                        tensor_id = id(input_tensor)
                        if tensor_id in tensor_grads:
                            # Use backend directly for accumulation
                            from titan_ai.tensor.core.tensor import Tensor
                            existing_grad = tensor_grads[tensor_id]
                            new_grad_backend = existing_grad._backend.add(input_grad._backend, existing_grad.dtype)
                            tensor_grads[tensor_id] = Tensor._from_backend(new_grad_backend, requires_grad=False)
                        else:
                            tensor_grads[tensor_id] = input_grad
    finally:
        # Restore previous tracking state
        if was_tracking:
            enable_tracking()


def _validate_initial_gradient(tensor: "Tensor", grad: "Tensor | None") -> "Tensor":
    """Validate and return the initial gradient for backward pass.

    Args:
        tensor: The output tensor.
        grad: Optional initial gradient.

    Returns:
        The validated initial gradient tensor.

    Raises:
        TensorValidationError: If validation fails.
    """
    if grad is None:
        if tensor.shape == ():
            # Scalar tensor: use 1.0 as initial gradient
            from titan_ai.tensor.backend.numpy_backend import NumpyBackend
            from titan_ai.tensor.core.tensor import Tensor
            grad_backend = NumpyBackend.ones((), tensor.dtype, tensor.device)
            return Tensor._from_backend(grad_backend, requires_grad=False)
        else:
            raise TensorValidationError(
                "backward() called on non-scalar tensor without providing "
                "an initial gradient. Use tensor.backward(grad) instead."
            )

    # Validate gradient shape
    if grad.shape != tensor.shape:
        raise TensorValidationError(
            f"Gradient shape {grad.shape} is incompatible with "
            f"tensor shape {tensor.shape}."
        )

    # Ensure gradient doesn't require grad
    if grad.requires_grad:
        from titan_ai.tensor.core.tensor import Tensor
        return Tensor._from_backend(grad._backend, requires_grad=False)

    return grad


def _build_topological_order(tensor: "Tensor") -> list:
    """Build topological order of computation graph starting from tensor.
    
    Args:
        tensor: The output tensor.
        
    Returns:
        List of AutogradNodes in topological order.
    """
    visited = set()
    order = []
    
    def _visit(node) -> None:
        if node in visited:
            return
        visited.add(node)
        
        # Visit all input nodes first
        for input_tensor in node.inputs:
            if input_tensor is not None and input_tensor.grad_fn is not None:
                _visit(input_tensor.grad_fn)
        
        order.append(node)
    
    if tensor.grad_fn is not None:
        _visit(tensor.grad_fn)
    
    return order
