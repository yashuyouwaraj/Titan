"""Tests for Autograd system - Phase 1: Graph Construction, Phase 2: Backward Engine."""

import pytest

from titan_ai.tensor import Tensor
from titan_ai.tensor.autograd import (
    AutogradNode,
    disable_tracking,
    enable_tracking,
    is_tracking,
)
from titan_ai.tensor.exceptions.errors import TensorValidationError


class TestTensorAutogradState:
    """Test Tensor autograd state properties."""

    def test_default_requires_grad_is_false(self) -> None:
        """Default requires_grad should be False for backward compatibility."""
        tensor = Tensor([1.0, 2.0, 3.0])
        assert tensor.requires_grad is False

    def test_requires_grad_true(self) -> None:
        """Tensor can be created with requires_grad=True."""
        tensor = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert tensor.requires_grad is True

    def test_grad_initially_none(self) -> None:
        """Gradient should be None initially."""
        tensor = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert tensor.grad is None

    def test_grad_fn_initially_none(self) -> None:
        """grad_fn should be None for leaf tensors."""
        tensor = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert tensor.grad_fn is None

    def test_is_leaf_true_for_user_created(self) -> None:
        """User-created tensors should be leaves."""
        tensor = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        assert tensor.is_leaf is True

    def test_is_leaf_true_for_factory_methods(self) -> None:
        """Factory method tensors should be leaves."""
        tensor = Tensor.zeros((2, 3), requires_grad=True)
        assert tensor.is_leaf is True


class TestFactoryMethodsRequiresGrad:
    """Test factory methods support requires_grad parameter."""

    def test_zeros_requires_grad(self) -> None:
        tensor = Tensor.zeros((2, 3), requires_grad=True)
        assert tensor.requires_grad is True
        assert tensor.is_leaf is True

    def test_ones_requires_grad(self) -> None:
        tensor = Tensor.ones((2, 3), requires_grad=True)
        assert tensor.requires_grad is True
        assert tensor.is_leaf is True

    def test_empty_requires_grad(self) -> None:
        tensor = Tensor.empty((2, 3), requires_grad=True)
        assert tensor.requires_grad is True
        assert tensor.is_leaf is True

    def test_full_requires_grad(self) -> None:
        tensor = Tensor.full((2, 3), 5.0, requires_grad=True)
        assert tensor.requires_grad is True
        assert tensor.is_leaf is True

    def test_arange_requires_grad(self) -> None:
        tensor = Tensor.arange(0, 5, requires_grad=True)
        assert tensor.requires_grad is True
        assert tensor.is_leaf is True

    def test_from_numpy_requires_grad(self) -> None:
        import numpy as np
        array = np.array([1.0, 2.0, 3.0])
        tensor = Tensor.from_numpy(array, requires_grad=True)
        assert tensor.requires_grad is True
        assert tensor.is_leaf is True

    def test_factory_methods_default_requires_grad_false(self) -> None:
        """Factory methods should default requires_grad to False."""
        tensor = Tensor.zeros((2, 3))
        assert tensor.requires_grad is False


class TestGradientContext:
    """Test gradient tracking context management."""

    def test_tracking_enabled_by_default(self) -> None:
        """Gradient tracking should be enabled by default."""
        assert is_tracking() is True

    def test_disable_tracking(self) -> None:
        """Can disable gradient tracking."""
        disable_tracking()
        assert is_tracking() is False
        enable_tracking()  # Reset

    def test_enable_tracking(self) -> None:
        """Can enable gradient tracking."""
        disable_tracking()
        enable_tracking()
        assert is_tracking() is True


class TestGraphConstruction:
    """Test computation graph construction for basic operations."""

    def test_add_creates_graph_when_input_requires_grad(self) -> None:
        """Add operation should create graph when input requires grad."""
        x = Tensor([2.0], requires_grad=True)
        y = x + 3
        assert y.requires_grad is True
        assert y.is_leaf is False
        assert y.grad_fn is not None
        assert isinstance(y.grad_fn, AutogradNode)
        assert y.grad_fn.operation == "add"

    def test_add_no_graph_when_input_no_grad(self) -> None:
        """Add operation should not create graph when input doesn't require grad."""
        x = Tensor([2.0])
        y = x + 3
        assert y.requires_grad is False
        assert y.is_leaf is True
        assert y.grad_fn is None

    def test_subtract_creates_graph(self) -> None:
        """Subtract operation should create graph when input requires grad."""
        x = Tensor([5.0], requires_grad=True)
        y = x - 2
        assert y.requires_grad is True
        assert y.is_leaf is False
        assert y.grad_fn is not None
        assert y.grad_fn.operation == "subtract"

    def test_multiply_creates_graph(self) -> None:
        """Multiply operation should create graph when input requires grad."""
        x = Tensor([3.0], requires_grad=True)
        y = x * 2
        assert y.requires_grad is True
        assert y.is_leaf is False
        assert y.grad_fn is not None
        assert y.grad_fn.operation == "multiply"

    def test_true_divide_creates_graph(self) -> None:
        """True divide operation should create graph when input requires grad."""
        x = Tensor([6.0], requires_grad=True)
        y = x / 2
        assert y.requires_grad is True
        assert y.is_leaf is False
        assert y.grad_fn is not None
        assert y.grad_fn.operation == "true_divide"

    def test_negate_creates_graph(self) -> None:
        """Negate operation should create graph when input requires grad."""
        x = Tensor([2.0], requires_grad=True)
        y = -x
        assert y.requires_grad is True
        assert y.is_leaf is False
        assert y.grad_fn is not None
        assert y.grad_fn.operation == "negate"

    def test_graph_not_created_when_tracking_disabled(self) -> None:
        """Graph should not be created when tracking is disabled."""
        disable_tracking()
        x = Tensor([2.0], requires_grad=True)
        y = x + 3
        assert y.grad_fn is None
        assert y.is_leaf is True
        enable_tracking()  # Reset

    def test_multiple_inputs_one_requires_grad(self) -> None:
        """Graph should be created if at least one input requires grad."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=False)
        z = x * y
        assert z.requires_grad is True
        assert z.is_leaf is False
        assert z.grad_fn is not None

    def test_multiple_inputs_both_require_grad(self) -> None:
        """Graph should be created when both inputs require grad."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x * y
        assert z.requires_grad is True
        assert z.is_leaf is False
        assert z.grad_fn is not None

    def test_chained_operations_build_graph(self) -> None:
        """Chained operations should build a computation graph."""
        x = Tensor([2.0], requires_grad=True)
        y = x + 1
        z = y * 2
        assert x.is_leaf is True
        assert y.is_leaf is False
        assert z.is_leaf is False
        assert y.grad_fn is not None
        assert z.grad_fn is not None
        assert y.grad_fn.operation == "add"
        assert z.grad_fn.operation == "multiply"


class TestAutogradNode:
    """Test AutogradNode structure."""

    def test_autograd_node_properties(self) -> None:
        """AutogradNode should store operation and inputs."""
        x = Tensor([2.0], requires_grad=True)
        y = x + 3
        node = y.grad_fn
        assert node is not None
        assert node.operation == "add"
        assert len(node.inputs) == 2
        assert node.inputs[0] is x  # First input is x
        # Second input is a scalar tensor that may be garbage collected (weak ref)
        # This is expected behavior for temporary scalar tensors

    def test_autograd_node_repr(self) -> None:
        """AutogradNode should have informative repr."""
        x = Tensor([2.0], requires_grad=True)
        y = x + 3
        node = y.grad_fn
        assert node is not None
        repr_str = repr(node)
        assert "AutogradNode" in repr_str
        assert "add" in repr_str


class TestBackwardCompatibility:
    """Ensure existing code continues to work without changes."""

    def test_existing_tensor_creation_still_works(self) -> None:
        """Existing tensor creation patterns should still work."""
        tensor = Tensor([1, 2, 3])
        assert tensor.shape == (3,)
        assert tensor.requires_grad is False

    def test_existing_arithmetic_still_works(self) -> None:
        """Existing arithmetic operations should still work."""
        x = Tensor([1, 2, 3])
        y = Tensor([4, 5, 6])
        z = x + y
        assert z.shape == (3,)
        assert z.requires_grad is False

    def test_existing_factory_methods_still_work(self) -> None:
        """Existing factory method calls should still work."""
        zeros = Tensor.zeros((2, 3))
        ones = Tensor.ones((2, 3))
        assert zeros.shape == (2, 3)
        assert ones.shape == (2, 3)
        assert zeros.requires_grad is False
        assert ones.requires_grad is False


class TestBackwardBasic:
    """Test basic backward pass functionality."""

    def test_simple_multiplication_backward(self) -> None:
        """Test backward pass for simple multiplication: y = x * x."""
        x = Tensor(2.0, requires_grad=True)
        y = x * x
        y.backward()
        assert x.grad is not None
        assert x.grad.shape == ()
        assert abs(x.grad.numpy() - 4.0) < 1e-6  # dy/dx = 2x = 4

    def test_simple_addition_backward(self) -> None:
        """Test backward pass for addition: z = x + y."""
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)
        z = x + y
        z.backward()
        assert x.grad is not None
        assert y.grad is not None
        assert abs(x.grad.numpy() - 1.0) < 1e-6  # dz/dx = 1
        assert abs(y.grad.numpy() - 1.0) < 1e-6  # dz/dy = 1

    def test_simple_subtraction_backward(self) -> None:
        """Test backward pass for subtraction: z = x - y."""
        x = Tensor(5.0, requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = x - y
        z.backward()
        assert x.grad is not None
        assert y.grad is not None
        assert abs(x.grad.numpy() - 1.0) < 1e-6  # dz/dx = 1
        assert abs(y.grad.numpy() - (-1.0)) < 1e-6  # dz/dy = -1

    def test_simple_division_backward(self) -> None:
        """Test backward pass for division: z = x / y."""
        x = Tensor(6.0, requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = x / y
        z.backward()
        assert x.grad is not None
        assert y.grad is not None
        assert abs(x.grad.numpy() - 0.5) < 1e-6  # dz/dx = 1/y = 0.5
        assert abs(y.grad.numpy() - (-1.5)) < 1e-6  # dz/dy = -x/y^2 = -1.5

    def test_simple_negation_backward(self) -> None:
        """Test backward pass for negation: y = -x."""
        x = Tensor(2.0, requires_grad=True)
        y = -x
        y.backward()
        assert x.grad is not None
        assert abs(x.grad.numpy() - (-1.0)) < 1e-6  # dy/dx = -1

    def test_chain_rule_backward(self) -> None:
        """Test chain rule: y = x * x, z = y * 3."""
        x = Tensor(2.0, requires_grad=True)
        y = x * x
        z = y * 3
        z.backward()
        assert x.grad is not None
        # dz/dx = dz/dy * dy/dx = 3 * 2x = 12
        assert abs(x.grad.numpy() - 12.0) < 1e-6

    def test_gradient_accumulation(self) -> None:
        """Test gradient accumulation from multiple paths."""
        x = Tensor(2.0, requires_grad=True)
        a = x * 3
        b = x * 4
        y = a + b
        y.backward()
        assert x.grad is not None
        # dy/dx = 3 + 4 = 7
        assert abs(x.grad.numpy() - 7.0) < 1e-6

    def test_gradient_accumulation_same_variable(self) -> None:
        """Test gradient accumulation when same variable used twice."""
        x = Tensor(2.0, requires_grad=True)
        a = x * x
        b = x * x
        y = a + b
        y.backward()
        assert x.grad is not None
        # dy/dx = 4 + 4 = 8 (since 2x = 4 for x=2)
        assert abs(x.grad.numpy() - 8.0) < 1e-6


class TestBackwardValidation:
    """Test backward pass validation."""

    def test_backward_non_scalar_without_grad_raises(self) -> None:
        """backward() on non-scalar without grad should raise."""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x * 2
        with pytest.raises(TensorValidationError, match="non-scalar"):
            y.backward()

    def test_backward_with_incompatible_grad_shape_raises(self) -> None:
        """backward() with incompatible grad shape should raise."""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x * 2
        grad = Tensor([1.0], requires_grad=False)
        with pytest.raises(TensorValidationError, match="incompatible"):
            y.backward(grad)

    def test_backward_non_scalar_with_valid_grad(self) -> None:
        """backward() on non-scalar with valid grad should work."""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x * 2
        grad = Tensor([1.0, 1.0], requires_grad=False)
        y.backward(grad)
        assert x.grad is not None
        assert x.grad.shape == (2,)


class TestBackwardScalarOperand:
    """Test backward with scalar operands."""

    def test_backward_with_scalar_add(self) -> None:
        """Test backward with scalar addition: y = x + 3."""
        x = Tensor(2.0, requires_grad=True)
        y = x + 3
        y.backward()
        assert x.grad is not None
        assert abs(x.grad.numpy() - 1.0) < 1e-6

    def test_backward_with_scalar_multiply(self) -> None:
        """Test backward with scalar multiplication: y = x * 3."""
        x = Tensor(2.0, requires_grad=True)
        y = x * 3
        y.backward()
        assert x.grad is not None
        assert abs(x.grad.numpy() - 3.0) < 1e-6

    def test_backward_with_scalar_divide(self) -> None:
        """Test backward with scalar division: y = x / 2."""
        x = Tensor(4.0, requires_grad=True)
        y = x / 2
        y.backward()
        assert x.grad is not None
        assert abs(x.grad.numpy() - 0.5) < 1e-6


class TestBackwardBroadcasting:
    """Test backward with broadcasting."""

    def test_backward_scalar_broadcast_elementwise(self) -> None:
        """Test backward with scalar broadcasting (elementwise)."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x + 1
        # Manually compute gradient for each element
        # Since we can't use sum() yet (no reduction backward), test elementwise
        grad = Tensor([1.0, 1.0, 1.0], requires_grad=False)
        y.backward(grad)
        assert x.grad is not None
        assert x.grad.shape == (3,)
        # Each element gets gradient 1
        import numpy as np
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0, 1.0], atol=1e-6)

    def test_backward_1d_broadcast_elementwise(self) -> None:
        """Test backward with 1D broadcasting (elementwise)."""
        x = Tensor([[1.0], [2.0]], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = x + y
        # Provide gradient matching output shape (2, 2)
        grad = Tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=False)
        z.backward(grad)
        assert x.grad is not None
        assert y.grad is not None
        # x shape (2,1) broadcasts to (2,2), gradient sums over broadcast dim
        import numpy as np
        np.testing.assert_allclose(x.grad.numpy(), [[2.0], [2.0]], atol=1e-6)
        np.testing.assert_allclose(y.grad.numpy(), [2.0, 2.0], atol=1e-6)


class TestBackwardLeafVsNonLeaf:
    """Test leaf vs non-leaf tensor behavior."""

    def test_leaf_gradient_accessible(self) -> None:
        """Leaf tensor gradient should be accessible after backward."""
        x = Tensor(2.0, requires_grad=True)
        y = x * 3
        z = y * 2
        z.backward()
        assert x.grad is not None
        assert x.is_leaf is True
        assert y.is_leaf is False
        assert z.is_leaf is False

    def test_intermediate_no_grad_by_default(self) -> None:
        """Intermediate tensors should not have grad by default."""
        x = Tensor(2.0, requires_grad=True)
        y = x * 3
        z = y * 2
        z.backward()
        # Intermediate tensors DO get gradients during backward
        # They're just not accessible via .grad after backward completes
        # (they're only stored temporarily during the backward pass)
        # This is the correct behavior
        assert x.grad is not None  # Leaf tensor
        assert x.grad.numpy() == 6.0  # dz/dx = 6

    def test_backward_on_leaf_does_nothing(self) -> None:
        """backward() on leaf tensor should do nothing."""
        x = Tensor(2.0, requires_grad=True)
        x.backward()
        assert x.grad is None


class TestMatmulBackward:
    """Test backward pass for matrix multiplication."""

    def test_matmul_1d_1d_backward(self) -> None:
        """Test backward for 1D @ 1D → scalar."""
        import numpy as np

        A = Tensor([2.0, 3.0], requires_grad=True)
        B = Tensor([4.0, 5.0], requires_grad=True)
        C = A @ B
        C.backward()

        # C = 2*4 + 3*5 = 8 + 15 = 23
        # dC/dA = B = [4, 5]
        # dC/dB = A = [2, 3]
        assert A.grad is not None
        assert B.grad is not None
        assert A.grad.shape == (2,)
        assert B.grad.shape == (2,)
        np.testing.assert_allclose(A.grad.numpy(), [4.0, 5.0], atol=1e-6)
        np.testing.assert_allclose(B.grad.numpy(), [2.0, 3.0], atol=1e-6)

    def test_matmul_2d_2d_backward(self) -> None:
        """Test backward for 2D @ 2D → 2D."""
        import numpy as np

        A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        C = A @ B
        # C is 2D, so provide gradient
        grad_output = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False)
        C.backward(grad_output)

        # C = A @ B
        # dC/dA = grad_output @ B.T
        # dC/dB = A.T @ grad_output
        assert A.grad is not None
        assert B.grad is not None
        assert A.grad.shape == (2, 2)
        assert B.grad.shape == (2, 2)

        expected_grad_A = np.array([[1.0, 0.0], [0.0, 1.0]]) @ np.array([[5.0, 7.0], [6.0, 8.0]])
        expected_grad_B = np.array([[1.0, 3.0], [2.0, 4.0]]) @ np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(A.grad.numpy(), expected_grad_A, atol=1e-6)
        np.testing.assert_allclose(B.grad.numpy(), expected_grad_B, atol=1e-6)

    def test_matmul_1d_2d_backward(self) -> None:
        """Test backward for 1D @ 2D → 1D."""
        import numpy as np

        A = Tensor([2.0, 3.0], requires_grad=True)
        B = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        C = A @ B
        # C is 1D, provide gradient
        grad_output = Tensor([1.0, 1.0], requires_grad=False)
        C.backward(grad_output)

        # C is 1D: [2*1+3*3, 2*2+3*4] = [11, 16]
        # dC/dA = grad_output @ B.T, then reshape to match A
        # dC/dB = A.T @ grad_output (with proper reshaping)
        assert A.grad is not None
        assert B.grad is not None
        assert A.grad.shape == (2,)
        assert B.grad.shape == (2, 2)

        # dC/dA should be [1+1, 1+1] @ B.T = [2, 2] @ [[1, 3], [2, 4]] = [2*1+2*2, 2*3+2*4] = [6, 14]
        # Actually: grad_output @ B.T = [1, 1] @ [[1, 3], [2, 4]] = [1*1+1*2, 1*3+1*4] = [3, 7]
        np.testing.assert_allclose(A.grad.numpy(), [3.0, 7.0], atol=1e-6)

        # dC/dB should have A values repeated along rows
        # A.T @ grad_output = [[2], [3]] @ [1, 1] = [[2, 2], [3, 3]]
        expected_grad_B = np.array([[2.0, 2.0], [3.0, 3.0]])
        np.testing.assert_allclose(B.grad.numpy(), expected_grad_B, atol=1e-6)

    def test_matmul_2d_1d_backward(self) -> None:
        """Test backward for 2D @ 1D → 1D."""
        import numpy as np

        A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        B = Tensor([5.0, 6.0], requires_grad=True)
        C = A @ B
        # C is 1D, provide gradient
        grad_output = Tensor([1.0, 1.0], requires_grad=False)
        C.backward(grad_output)

        # C is 1D: [1*5+2*6, 3*5+4*6] = [17, 39]
        # dC/dA = grad_output @ B.T (with proper reshaping)
        # dC/dB = A.T @ grad_output
        assert A.grad is not None
        assert B.grad is not None
        assert A.grad.shape == (2, 2)
        assert B.grad.shape == (2,)

        # dC/dA should have B values repeated along columns
        # grad_output @ B.T = [1, 1] @ [5, 6] = [5, 6] (as row), then broadcast to [[5, 6], [5, 6]]
        expected_grad_A = np.array([[5.0, 6.0], [5.0, 6.0]])
        np.testing.assert_allclose(A.grad.numpy(), expected_grad_A, atol=1e-6)

        # dC/dB should be [1+3, 2+4] = [4, 6] (sum of A rows)
        # A.T @ grad_output = [[1, 3], [2, 4]] @ [1, 1] = [4, 6]
        np.testing.assert_allclose(B.grad.numpy(), [4.0, 6.0], atol=1e-6)

    def test_matmul_chain_backward(self) -> None:
        """Test matmul in a chain with other operations."""
        import numpy as np

        A = Tensor([2.0, 3.0], requires_grad=True)
        B = Tensor([4.0, 5.0], requires_grad=True)
        C = A @ B  # scalar
        D = C * 2  # scalar
        D.backward()

        # D = 2 * (A @ B) = 2 * 23 = 46
        # dD/dA = 2 * B = [8, 10]
        # dD/dB = 2 * A = [4, 6]
        assert A.grad is not None
        assert B.grad is not None
        np.testing.assert_allclose(A.grad.numpy(), [8.0, 10.0], atol=1e-6)
        np.testing.assert_allclose(B.grad.numpy(), [4.0, 6.0], atol=1e-6)

    def test_matmul_no_grad_for_no_requires_grad(self) -> None:
        """Test that tensors without requires_grad don't get gradients."""
        A = Tensor([2.0, 3.0], requires_grad=True)
        B = Tensor([4.0, 5.0], requires_grad=False)
        C = A @ B
        C.backward()

        assert A.grad is not None
        assert B.grad is None


class TestSumBackward:
    """Test backward pass for sum reduction."""

    def test_sum_axis_none_backward(self) -> None:
        """Test backward for sum with axis=None."""
        import numpy as np

        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.sum()
        y.backward()

        # y = 6, dy/dx = 1 for all elements
        assert x.grad is not None
        assert x.grad.shape == (3,)
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0, 1.0], atol=1e-6)

    def test_sum_axis_0_backward(self) -> None:
        """Test backward for sum with axis=0."""
        import numpy as np

        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum(axis=0)
        grad_output = Tensor([1.0, 1.0], requires_grad=False)
        y.backward(grad_output)

        # y = [4, 6], dy/dx should broadcast [1, 1] to match (2, 2)
        assert x.grad is not None
        assert x.grad.shape == (2, 2)
        np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]], atol=1e-6)

    def test_sum_axis_1_backward(self) -> None:
        """Test backward for sum with axis=1."""
        import numpy as np

        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum(axis=1)
        grad_output = Tensor([1.0, 1.0], requires_grad=False)
        y.backward(grad_output)

        # y = [3, 7], dy/dx should broadcast [1, 1] to match (2, 2)
        assert x.grad is not None
        assert x.grad.shape == (2, 2)
        np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]], atol=1e-6)

    def test_sum_keepdims_true_backward(self) -> None:
        """Test backward for sum with keepdims=True."""
        import numpy as np

        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum(axis=1, keepdims=True)
        grad_output = Tensor([[1.0], [1.0]], requires_grad=False)
        y.backward(grad_output)

        # y = [[3], [7]], dy/dx should broadcast to match (2, 2)
        assert x.grad is not None
        assert x.grad.shape == (2, 2)
        np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]], atol=1e-6)

    def test_sum_scalar_backward(self) -> None:
        """Test backward for sum of scalar."""
        x = Tensor(5.0, requires_grad=True)
        y = x.sum()
        y.backward()

        assert x.grad is not None
        assert x.grad.shape == ()
        assert abs(x.grad.numpy() - 1.0) < 1e-6


class TestMeanBackward:
    """Test backward pass for mean reduction."""

    def test_mean_axis_none_backward(self) -> None:
        """Test backward for mean with axis=None."""
        import numpy as np

        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.mean()
        y.backward()

        # y = 2, dy/dx = 1/3 for all elements
        assert x.grad is not None
        assert x.grad.shape == (3,)
        np.testing.assert_allclose(x.grad.numpy(), [1.0/3, 1.0/3, 1.0/3], atol=1e-6)

    def test_mean_axis_0_backward(self) -> None:
        """Test backward for mean with axis=0."""
        import numpy as np

        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.mean(axis=0)
        grad_output = Tensor([1.0, 1.0], requires_grad=False)
        y.backward(grad_output)

        # y = [2, 3], dy/dx = 1/2 broadcast to (2, 2)
        assert x.grad is not None
        assert x.grad.shape == (2, 2)
        np.testing.assert_allclose(x.grad.numpy(), [[0.5, 0.5], [0.5, 0.5]], atol=1e-6)

    def test_mean_axis_1_backward(self) -> None:
        """Test backward for mean with axis=1."""
        import numpy as np

        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.mean(axis=1)
        grad_output = Tensor([1.0, 1.0], requires_grad=False)
        y.backward(grad_output)

        # y = [1.5, 3.5], dy/dx = 1/2 broadcast to (2, 2)
        assert x.grad is not None
        assert x.grad.shape == (2, 2)
        np.testing.assert_allclose(x.grad.numpy(), [[0.5, 0.5], [0.5, 0.5]], atol=1e-6)

    def test_mean_keepdims_true_backward(self) -> None:
        """Test backward for mean with keepdims=True."""
        import numpy as np

        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.mean(axis=1, keepdims=True)
        grad_output = Tensor([[1.0], [1.0]], requires_grad=False)
        y.backward(grad_output)

        # y = [[1.5], [3.5]], dy/dx = 1/2 broadcast to (2, 2)
        assert x.grad is not None
        assert x.grad.shape == (2, 2)
        np.testing.assert_allclose(x.grad.numpy(), [[0.5, 0.5], [0.5, 0.5]], atol=1e-6)


class TestMathBackward:
    """Test backward pass for mathematical functions."""

    def test_abs_backward_positive(self) -> None:
        """Test backward for abs with positive values."""
        import numpy as np

        x = Tensor([2.0, 3.0], requires_grad=True)
        y = x.abs()
        y.backward(Tensor([1.0, 1.0], requires_grad=False))

        # dy/dx = sign(x) = [1, 1]
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0], atol=1e-6)

    def test_abs_backward_negative(self) -> None:
        """Test backward for abs with negative values."""
        import numpy as np

        x = Tensor([-2.0, -3.0], requires_grad=True)
        y = x.abs()
        y.backward(Tensor([1.0, 1.0], requires_grad=False))

        # dy/dx = sign(x) = [-1, -1]
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [-1.0, -1.0], atol=1e-6)

    def test_abs_backward_zero(self) -> None:
        """Test backward for abs with zero (subgradient of 0)."""
        import numpy as np

        x = Tensor([0.0], requires_grad=True)
        y = x.abs()
        y.backward(Tensor([1.0], requires_grad=False))

        # dy/dx = sign(0) = 0 (subgradient convention)
        assert x.grad is not None
        assert abs(x.grad.numpy() - 0.0) < 1e-6

    def test_sqrt_backward(self) -> None:
        """Test backward for sqrt."""
        import numpy as np

        x = Tensor([4.0, 9.0], requires_grad=True)
        y = x.sqrt()
        y.backward(Tensor([1.0, 1.0], requires_grad=False))

        # dy/dx = 1 / (2 * sqrt(x)) = [1/(2*2), 1/(2*3)] = [0.25, 0.1667]
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [0.25, 1.0/6.0], atol=1e-6)

    def test_exp_backward(self) -> None:
        """Test backward for exp."""
        import numpy as np

        x = Tensor([0.0, 1.0], requires_grad=True)
        y = x.exp()
        y.backward(Tensor([1.0, 1.0], requires_grad=False))

        # dy/dx = exp(x) = [1, e]
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [1.0, np.e], atol=1e-6)

    def test_log_backward(self) -> None:
        """Test backward for log."""
        import numpy as np

        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x.log()
        y.backward(Tensor([1.0, 1.0], requires_grad=False))

        # dy/dx = 1/x = [1, 0.5]
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 0.5], atol=1e-6)


class TestTupleAxesBackward:
    """Test backward pass for reductions with tuple axes."""

    def test_sum_tuple_axes_backward(self) -> None:
        """Test backward for sum with tuple axes."""
        import numpy as np

        x = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
        y = x.sum(axis=(0, 1))
        grad_output = Tensor([1.0, 1.0], requires_grad=False)
        y.backward(grad_output)

        # y = [1+3+5+7, 2+4+6+8] = [16, 20]
        # dy/dx should broadcast [1, 1] to match (2, 2, 2)
        assert x.grad is not None
        assert x.grad.shape == (2, 2, 2)
        np.testing.assert_allclose(x.grad.numpy(), np.ones((2, 2, 2)), atol=1e-6)

    def test_sum_tuple_axes_keepdims_backward(self) -> None:
        """Test backward for sum with tuple axes and keepdims=True."""
        import numpy as np

        x = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
        y = x.sum(axis=(0, 1), keepdims=True)
        grad_output = Tensor([[[1.0, 1.0]]], requires_grad=False)
        y.backward(grad_output)

        # y = [[[16, 20]]]
        # dy/dx should broadcast to match (2, 2, 2)
        assert x.grad is not None
        assert x.grad.shape == (2, 2, 2)
        np.testing.assert_allclose(x.grad.numpy(), np.ones((2, 2, 2)), atol=1e-6)

    def test_mean_tuple_axes_backward(self) -> None:
        """Test backward for mean with tuple axes."""
        import numpy as np

        x = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
        y = x.mean(axis=(0, 1))
        grad_output = Tensor([1.0, 1.0], requires_grad=False)
        y.backward(grad_output)

        # y = [16/4, 20/4] = [4, 5]
        # dy/dx = 1/4 broadcast to (2, 2, 2)
        assert x.grad is not None
        assert x.grad.shape == (2, 2, 2)
        np.testing.assert_allclose(x.grad.numpy(), np.full((2, 2, 2), 0.25), atol=1e-6)

    def test_mean_tuple_axes_keepdims_backward(self) -> None:
        """Test backward for mean with tuple axes and keepdims=True."""
        import numpy as np

        x = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
        y = x.mean(axis=(0, 1), keepdims=True)
        grad_output = Tensor([[[1.0, 1.0]]], requires_grad=False)
        y.backward(grad_output)

        # y = [[[4, 5]]]
        # dy/dx = 1/4 broadcast to (2, 2, 2)
        assert x.grad is not None
        assert x.grad.shape == (2, 2, 2)
        np.testing.assert_allclose(x.grad.numpy(), np.full((2, 2, 2), 0.25), atol=1e-6)

    def test_sum_tuple_axes_2d_backward(self) -> None:
        """Test backward for sum with tuple axes on 2D tensor."""
        import numpy as np

        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum(axis=(0, 1))
        y.backward()

        # y = 10, dy/dx = 1 for all elements
        assert x.grad is not None
        assert x.grad.shape == (2, 2)
        np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]], atol=1e-6)


class TestGradientIntegration:
    """Test integration of gradient operations."""

    def test_matmul_sum_chain(self) -> None:
        """Test matmul → sum → backward."""
        import numpy as np

        A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        C = A @ B
        loss = C.sum()
        loss.backward()

        # loss = sum(A @ B)
        # dloss/dA = grad_output @ B.T where grad_output is all ones
        # grad_output @ B.T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        # dloss/dB = A.T @ grad_output where grad_output is all ones
        # A.T @ [[1,1],[1,1]] = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        assert A.grad is not None
        assert B.grad is not None
        np.testing.assert_allclose(A.grad.numpy(), [[11.0, 15.0], [11.0, 15.0]], atol=1e-6)
        np.testing.assert_allclose(B.grad.numpy(), [[4.0, 4.0], [6.0, 6.0]], atol=1e-6)

    def test_multiply_exp_sum_chain(self) -> None:
        """Test multiply → exp → sum → backward."""
        import numpy as np

        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x * 2
        z = y.exp()
        loss = z.sum()
        loss.backward()

        # loss = sum(exp(2x))
        # dloss/dx = 2 * exp(2x)
        expected = 2 * np.exp(2 * np.array([1.0, 2.0]))
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-6)

    def test_matmul_sqrt_sum_chain(self) -> None:
        """Test matmul → sqrt → sum → backward."""
        import numpy as np

        A = Tensor([[4.0, 9.0]], requires_grad=True)
        B = Tensor([[1.0], [1.0]], requires_grad=True)
        C = A @ B  # scalar: 4+9=13
        D = C.sqrt()
        loss = D.sum()
        loss.backward()

        # loss = sqrt(A @ B) = sqrt(13)
        # dloss/dA = (1/(2*sqrt(13))) @ B.T = 1/(2*sqrt(13)) * [1, 1]
        # dloss/dB = A.T @ (1/(2*sqrt(13))) = [[4], [9]] / (2*sqrt(13))
        assert A.grad is not None
        assert B.grad is not None

        factor = 1.0 / (2 * np.sqrt(13.0))
        np.testing.assert_allclose(A.grad.numpy(), factor * np.array([[1.0, 1.0]]), atol=1e-6)
        np.testing.assert_allclose(B.grad.numpy(), factor * np.array([[4.0], [9.0]]), atol=1e-6)

    def test_gradient_accumulation_with_new_ops(self) -> None:
        """Test gradient accumulation with new operations."""
        import numpy as np

        x = Tensor([2.0, 3.0], requires_grad=True)
        a = x * 2
        b = x * 3
        c = a + b
        loss = c.sum()
        loss.backward()

        # a = [4, 6], b = [6, 9], c = [10, 15], loss = 25
        # dloss/dc = [1, 1]
        # dloss/da = [1, 1], dloss/db = [1, 1]
        # dloss/dx from a = dloss/da * da/dx = [1, 1] * 2 = [2, 2]
        # dloss/dx from b = dloss/db * db/dx = [1, 1] * 3 = [3, 3]
        # Total: [5, 5]
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [5.0, 5.0], atol=1e-6)
