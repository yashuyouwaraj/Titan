# Titan AI — Tensor Library

---

# Purpose

This document is the single authoritative reference for the Titan AI Tensor Library.

The Tensor Library provides Titan's owned numerical tensor abstraction — the foundation for autograd, neural networks, transformers, training, and inference. It defines the public API, backend architecture, dtype and device model, broadcasting semantics, and engineering standards for numerical correctness.

No other document should duplicate Tensor Library architecture details. Related systems (LLM, training, inference) reference this document rather than redefining tensor behavior.

---

# Scope

## In scope (Tensor Library)

- Tensor metadata: shape, dtype, device, ndim, size
- CPU tensor storage and operations (Week 1 subset)
- NumPy-backed CPU numerical backend
- Broadcasting semantics
- Error handling for invalid operations
- Numerical correctness validation against NumPy
- Testing and benchmarking foundations

## Out of scope (later phases)

- Autograd
- Neural network layers
- Transformer blocks
- Training loops
- Inference engines
- GPU/CUDA backends
- Distributed tensors
- Custom Triton kernels

---

# Why Titan AI Needs Its Own Tensor Abstraction

Titan AI is built from first principles with complete ownership of the intelligence stack. A proprietary or framework-coupled tensor layer would constrain architecture, device strategy, memory management, and future autograd design.

PyTorch may be used for research comparison or benchmarking where useful, but Titan's Tensor API must remain internally owned and not tightly coupled to PyTorch.

NumPy serves as the initial CPU numerical reference backend behind an explicit backend interface, enabling:

- Correctness validation against a mature reference
- A clear migration path to custom CPU and GPU backends
- Stable public API independent of any single framework

---

# Design Goals

- **Correctness first** — numerical accuracy over premature optimization during Week 1
- **Explicit metadata** — shape, dtype, device are always inspectable
- **Backend isolation** — storage and compute live behind a backend protocol
- **Replaceability** — CPU NumPy today; CUDA/Triton tomorrow without API breakage
- **Useful errors** — shape, dtype, device, and broadcasting failures are descriptive
- **Autograd-ready** — metadata and operation boundaries designed for future gradient tracking
- **Testability** — every operation verifiable against NumPy

---

# Public API

**Currently implemented (Day 2–4):**

```python
from titan_ai.tensor import Tensor, Device, Dtype

# Construction
tensor = Tensor([1, 2, 3])
tensor = Tensor([[1, 2], [3, 4]], dtype=Dtype.float32, device=Device.cpu())
tensor = Tensor(numpy_array, copy=True)

# Factories
Tensor.zeros((2, 3), dtype=Dtype.float32)
Tensor.ones((2, 3))
Tensor.empty((2, 3))
Tensor.full((2, 3), 7)
Tensor.arange(5)
Tensor.arange(2, 10, 2)
Tensor.from_numpy(numpy_array, copy=True)

# Metadata
tensor.shape
tensor.ndim
tensor.dtype
tensor.device
tensor.size

# Shape operations
tensor.reshape(2, 3)
tensor.reshape((2, -1))
tensor.transpose()
tensor.transpose(1, 0, 2)
tensor.T  # 2D only
tensor.flatten()
tensor.squeeze()
tensor.squeeze(axis=0)
tensor.unsqueeze(0)

# Indexing and slicing
tensor[0]
tensor[-1]
tensor[1:3]
tensor[:, 0]
tensor[1, 2]

# NumPy boundary
array = tensor.numpy()  # copy by default
view = tensor.numpy(copy=False)

# Indexing returns Tensor for array-like results, Python scalar for scalars

# Element-wise arithmetic (returns a new Tensor)
a + b
a + 5
5 + a
a - b
a - 5
5 - a
a * b  # element-wise, not matrix multiplication
a * 5
5 * a
a / b  # true division
a / 5
5 / a
-a

# Matrix multiplication
a @ b
```

**Planned (later Week 1 days):**

```python
tensor.sum(...)
tensor.mean(...)
tensor.exp()
```

The public import surface is `titan_ai.tensor`. Backend classes are not part of the primary public API.

---

# Factory Methods

**Status:** implemented (Day 3).

| Method | Description |
|--------|-------------|
| `Tensor.zeros` | Zero-filled tensor; default dtype `float64` |
| `Tensor.ones` | One-filled tensor; default dtype `float64` |
| `Tensor.empty` | Uninitialized allocation; contents not validated in tests |
| `Tensor.full` | Constant-filled tensor; dtype inferred from value if omitted |
| `Tensor.arange` | Arithmetic range; supports `stop` or `start, stop, step` |
| `Tensor.from_numpy` | Explicit NumPy conversion; same `copy` semantics as constructor |

All factories validate shape, dtype, and device through the backend boundary.

---

# Shape Operations

**Status:** implemented (Day 3).

| Operation | Behavior |
|-----------|----------|
| `reshape` | Supports tuple or integer args; single `-1` inference |
| `transpose` | Reverses axes when called without args; optional axis permutation |
| `T` | 2D transpose property |
| `flatten` | Returns 1D **copy** |
| `squeeze` | Removes singleton dimensions; optional axis |
| `unsqueeze` | Inserts singleton dimension; supports negative axes |

Shape operations preserve dtype and device. Invalid shapes and axes raise domain-specific errors.

---

# Indexing and Slicing

**Status:** implemented (Day 3).

- Integer indexing (including negative indices)
- Slices and step slices
- Multi-dimensional indexing with integers and slices
- Scalar tensors indexed with `tensor[()]`

**Return semantics:**

- Array-like results return a new `Tensor`
- Scalar extraction returns a Python `int`, `float`, or `bool`
- Raw NumPy arrays are not exposed from indexing

**Not implemented:** boolean masks, fancy indexing, advanced NumPy indexing forms.

---

# NumPy Boundary

**`Tensor.from_numpy`:** identical copy semantics to `Tensor(ndarray, copy=...)`. Default `copy=True`.

**`tensor.numpy(copy=True)`:** returns a NumPy array copy by default. `copy=False` returns a view sharing backend storage when supported.

NumPy remains behind the backend; these methods are explicit conversion boundaries.

---

# Memory and View Semantics (Day 3)

| Operation | Storage behavior |
|-----------|------------------|
| Constructor / `from_numpy` | `copy=True` copies; `copy=False` shares if dtype matches |
| `reshape` | May share storage (NumPy view when possible) |
| `transpose` | Typically a view |
| `flatten` | Always copies |
| `squeeze` / `unsqueeze` | Typically views |
| Slice indexing | View sharing storage with parent |
| Integer indexing | Scalar value, no tensor storage |
| `numpy(copy=True)` | Copy |
| `numpy(copy=False)` | View when backend supports it |
| Arithmetic (`+ - * /` and negation) | New storage; does not alias inputs |
| Matrix multiplication (`@`) | New storage; does not alias inputs |

These semantics are tested and matter for future autograd and performance work.

---

# Tensor Metadata

Every tensor owns explicit metadata independent of the backend implementation.

| Property | Description |
|----------|-------------|
| `shape` | Tuple of dimension sizes |
| `ndim` | Number of dimensions |
| `dtype` | Element type (`Dtype` enum) |
| `device` | Execution device (`Device` enum) |
| `size` | Total number of elements |

Metadata is owned by the `Tensor` wrapper and validated at construction and on operations that change shape or type.

---

# Backend Architecture

**Currently implemented (Day 2–4):**

```text
titan_ai/tensor/
    core/tensor.py          # Tensor class and operators
    backend/base.py         # TensorBackend ABC
    backend/numpy_backend.py
    dtypes/dtypes.py
    devices/devices.py
    exceptions/errors.py
    operations/shape.py
    operations/broadcast.py
    operations/promotion.py
    operations/arithmetic.py
    operations/matmul.py
```

The `Tensor` class holds a `TensorBackend` instance and exposes metadata through read-only properties. Public operators dispatch through operation modules to `TensorBackend`. `NumpyBackend` is the only concrete backend.

**Planned (later days):**

```text
    operations/reductions.py
    operations/math.py
```

Future GPU backends implement `TensorBackend` without changing the public `Tensor` API.

---

# CPU / NumPy Backend

**Status:** foundation implemented (Day 2); arithmetic and matmul primitives implemented (Day 4).

`NumpyBackend` stores data in a `numpy.ndarray` and implements `TensorBackend` metadata, shape, indexing, arithmetic, and matmul methods. NumPy is used only behind the backend boundary.

Construction from array-like data flows through `NumpyBackend.from_data`, which handles dtype conversion and copy semantics.

NumPy is a **backend dependency**, not the public API.

---

# Future GPU Backend Strategy

**Status:** not implemented; not part of Week 1.

Planned evolution:

1. Define `Device` with `cpu` (Week 1) and `cuda` (future)
2. Implement `CudaBackend` behind the same protocol as `NumpyBackend`
3. Operations dispatch on `tensor.device` to the appropriate backend
4. Optional Triton kernels for hot paths after correctness is established

CUDA support will not be claimed in documentation or API until implemented and tested.

---

# Device Abstraction

**Currently implemented (Day 2):**

- `Device` class with `cpu` support
- `Device.cpu()` factory
- Every tensor carries an explicit `device` property
- `Device("cuda")` raises `UnsupportedDeviceError` with a clear message
- Device equality supports `Device` instances and `"cpu"` strings

**Planned:** `cuda` when a GPU backend exists and is tested.

---

# Dtype Abstraction

**Currently implemented (Day 2):**

| Dtype | Status |
|-------|--------|
| `float32` | Implemented and tested |
| `float64` | Implemented and tested |
| `int32` | Implemented and tested |
| `int64` | Implemented and tested |
| `bool` | Implemented and tested |

`Dtype` is a controlled `Enum` with `to_numpy()` and `from_numpy()` conversion at the backend boundary. Unsupported NumPy dtypes raise `UnsupportedDtypeError`.

**Planned:** `float16`, `bfloat16` when GPU backends exist.

---

# Arithmetic Operators

**Status:** implemented (Day 4).

| Operator | Meaning |
|----------|---------|
| `a + b` / `__radd__` | Element-wise addition |
| `a - b` / `__rsub__` | Element-wise subtraction |
| `a * b` / `__rmul__` | Element-wise multiplication |
| `a / b` / `__rtruediv__` | Element-wise true division |
| `-a` | Unary negation |
| `a @ b` | Matrix multiplication |

Every operator returns a **new** `Tensor`. Operands are not mutated.

Supported scalar operands: Python `bool`, `int`, `float`, and NumPy scalars whose dtype maps to a Titan `Dtype`. Lists, strings, and other objects raise `UnsupportedOperandError`.

`*` is never matrix multiplication. Use `@` for matmul.

**Not implemented:** `__pow__`, `__mod__`, `__floordiv__`, comparisons, logical operators, in-place operators.

---

# Broadcasting

**Status:** implemented (Day 4) for element-wise arithmetic.

Rules (NumPy-compatible, validated before backend execution):

1. Compare dimensions from the trailing axis.
2. Dimensions are compatible when they are equal, or when either is `1`.
3. A scalar (shape `()` or a Python/NumPy scalar) broadcasts to any shape.
4. The output shape takes the non-`1` size in each aligned dimension.
5. Incompatible shapes raise `BroadcastError` with both operand shapes.

Examples:

| Operands | Result shape |
|----------|----------------|
| `(3,)` + `(3,)` | `(3,)` |
| `(2, 3)` + `(3,)` | `(2, 3)` |
| `(2, 3)` + `(1, 3)` | `(2, 3)` |
| `(2, 1)` + `(1, 4)` | `(2, 4)` |
| scalar + `(2, 3)` | `(2, 3)` |
| `(2, 3)` + `(2, 4)` | `BroadcastError` |

Broadcasting is implemented once in `operations/broadcast.py` and shared by add, subtract, multiply, and divide. Matrix multiplication does **not** use this mechanism.

---

# Dtype Promotion

**Status:** implemented (Day 4). Deterministic Titan rules for the five supported dtypes.

## Tensor / Tensor

| | bool | int32 | int64 | float32 | float64 |
|---|------|-------|-------|---------|---------|
| **bool** | bool | int32 | int64 | float32 | float64 |
| **int32** | int32 | int32 | int64 | float64 | float64 |
| **int64** | int64 | int64 | int64 | float64 | float64 |
| **float32** | float32 | float64 | float64 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 |

Mixed integer and `float32` promote to `float64` so integer magnitude is not rounded through `float32`. This matches NumPy `result_type` for this dtype subset.

## Tensor / Python scalar (weak scalars)

Python scalars do not force a wider dtype when the tensor already has a compatible kind:

- `int32 + 1` → `int32`
- `float32 + 1` → `float32`
- `float32 + 1.0` → `float32`
- `int32 + 1.0` → `float64`
- `bool + 1` → `int64`

## Tensor / NumPy scalar (strong scalars)

NumPy scalars use the Tensor/Tensor table. Example: `float32 + np.float64(1.0)` → `float64`.

## True division

After the arithmetic promotion above:

- If the result would be `bool`, `int32`, or `int64`, the division dtype is `float64`.
- `float32 / float32` remains `float32`.
- `float64` stays `float64`.

Integer true division therefore never silently truncates.

## Unary negation

Shape and device are preserved. `bool` tensors promote to `int64` (`True` → `-1`) because Titan does not provide NumPy's `int8` result for boolean negation.

---

# Device Compatibility

**Status:** implemented (Day 4) for CPU.

- Both operands of a Tensor/Tensor operation must be on the same device.
- Day 4 only supports `cpu`.
- Device mismatch raises `DeviceMismatchError`.
- An unsupported device raises `UnsupportedDeviceError`.
- Titan **never** silently moves data between devices.

---

# Matrix Multiplication

**Status:** implemented (Day 4) for 1D and 2D tensors.

| Operands | Result |
|----------|--------|
| `(K,) @ (K,)` | `()` scalar tensor (inner product) |
| `(M, K) @ (K, N)` | `(M, N)` |
| `(K,) @ (K, N)` | `(N,)` |
| `(M, K) @ (K,)` | `(M,)` |

Incompatible inner dimensions raise `InvalidShapeError` with both shapes. Rank 0 (scalar tensors) and rank > 2 are rejected; batched N-D matmul is planned, not implemented.

Boolean operands promote to `int64` before multiplication.

Matmul uses dedicated shape rules, not element-wise broadcasting.

---

# Arithmetic Memory Semantics

Arithmetic and matmul allocate **new** backend storage. Results do not alias operand storage. Changing an input array (for example through `copy=False` construction) does not change a previously computed result.

---

# NumPy Reference Strategy (Day 4)

NumPy is the CPU numerical reference, not the public API.

Tests compare Titan results to the equivalent NumPy expression for representative integers, `float32`, `float64`, scalars, vectors, matrices, and broadcasted shapes. Floating-point comparisons use dtype-appropriate tolerances.

This is **not** a claim of full NumPy API compatibility.

Floating-point division by zero follows NumPy: `inf` / `nan` values, typically with a NumPy runtime warning rather than a Titan exception.

---

# Memory Ownership Considerations

**Implemented behavior (Day 2):**

| Input | `copy=True` (default) | `copy=False` |
|-------|----------------------|--------------|
| Python list | New NumPy allocation | N/A (lists always copied) |
| NumPy array | New array copy | Shares underlying storage if dtype matches exactly |

**Rationale for default `copy=True`:** prevents accidental aliasing when users pass NumPy arrays that may be mutated elsewhere.

**Zero-copy (`copy=False`):** only valid for NumPy arrays whose dtype exactly matches the resolved tensor dtype. Dtype mismatch raises `TensorValidationError` rather than silently casting.

Backend owns raw storage; `Tensor` owns a reference to the backend instance. The `numpy_array` property on `NumpyBackend` is internal to the tensor package (used for tests and future operations), not public API.

---

# Future Autograd Compatibility

**Status:** not implemented.

Week 1 designs for future autograd without implementing it:

- Operations return new `Tensor` instances (no silent mutation of shared storage for arithmetic)
- Operation modules are discrete functions suitable for wrapping with gradient rules
- `Tensor` metadata remains on the wrapper, not scattered in backend-only state
- Backend protocol may later expose differentiable primitive hooks

Autograd belongs to a subsequent phase after the CPU tensor foundation is correct and tested.

---

# Error Handling Strategy

**Implemented (Day 2–4):**

| Exception | When |
|-----------|------|
| `TitanTensorError` | Base for all tensor errors |
| `TensorConstructionError` | Invalid or unconvertible input data |
| `TensorValidationError` | Validation failures |
| `UnsupportedDeviceError` | Unsupported device (including CUDA in Week 1) |
| `UnsupportedDtypeError` | Unsupported or unmapped dtype |
| `InvalidShapeError` | Invalid or incompatible shape, including matmul ranks/dims |
| `InvalidAxisError` | Invalid axis for shape operations |
| `TensorIndexError` | Invalid indexing |
| `BroadcastError` | Incompatible element-wise broadcast shapes |
| `DeviceMismatchError` | Tensor operands on different devices |
| `UnsupportedOperandError` | Operand type is not a Tensor or supported scalar |

Metadata properties (`shape`, `dtype`, `device`, `ndim`, `size`) are read-only Python properties without setters. Result metadata after arithmetic is derived from the backend result, not reconstructed independently.

---

# Numerical Correctness Strategy

1. Implement operation against NumPy backend
2. Compare results to direct NumPy computation in tests
3. Use `numpy.testing.assert_allclose` with dtype-appropriate tolerances
4. Cover scalars, vectors, matrices, higher-dimensional tensors, broadcasting, and edge cases
5. No performance claims without benchmarks

---

# Testing Strategy

**Currently implemented (Day 4):** 199 unit tests under `tests/ai/tensor/` covering construction, metadata, dtypes, devices, factories, shape ops, indexing, arithmetic, broadcasting, dtype promotion, matmul, exceptions, storage ownership, and public API imports.

**Planned coverage (later days):** reductions, mathematical functions.

---

# Benchmark Strategy

**Currently implemented:** directory placeholder (`benchmarks/tensor/`).

**Planned:**

- Baseline timings for creation, addition, matmul, reshape, reductions
- Record operation, tensor size, execution time, environment
- Optional NumPy comparison without unsupported performance claims
- Output to `benchmarks/output/` (gitignored)

Benchmarking establishes baselines before optimization, not premature tuning.

---

# Week 1 Scope

| Day | Focus |
|-----|-------|
| Day 1 | Repository setup, packaging, architecture documentation (this document) |
| Day 2 | Tensor core abstraction |
| Day 3 | Creation, shape, indexing, reshape, transpose |
| Day 4 | Arithmetic, broadcasting, matrix multiplication |
| Day 5 | Reductions, mathematical operations, numerical correctness |
| Day 6 | Tests, benchmarks, API cleanup, error handling |
| Day 7 | Documentation integration, final review, regression testing |

**Week 1 delivers:** CPU `Tensor` with the foundational operation subset, tests, and benchmarks.

---

# Explicit Non-Goals (Week 1)

- Autograd and backward pass
- Neural network modules
- Transformer implementation
- Training engine
- Inference engine
- GPU / CUDA / Triton
- Distributed tensors
- Full NumPy API parity
- PyTorch as runtime dependency

---

# Future Evolution

```text
Tensor Library (Week 1)
      ↓
Autograd Engine
      ↓
Neural Networks
      ↓
Transformer
      ↓
Training Pipeline
      ↓
Inference Engine
```

Each layer depends on a stable, tested tensor foundation.

---

# Related Documentation

- [`12_LLM_ARCHITECTURE.md`](12_LLM_ARCHITECTURE.md) — consumes tensor operations for model forward pass
- [`15_TRAINING_PIPELINE.md`](15_TRAINING_PIPELINE.md) — consumes tensors and future autograd
- [`16_INFERENCE_ENGINE.md`](16_INFERENCE_ENGINE.md) — consumes tensors at runtime
- [`07_TECH_STACK.md`](../../02_ENGINEERING/07_TECH_STACK.md) — NumPy as numerical foundation
- [`08_TECHNOLOGY_DECISIONS.md`](../../02_ENGINEERING/08_TECHNOLOGY_DECISIONS.md) — technology rationale
- [`10_ENGINEERING_CONSTITUTION.md`](../../02_ENGINEERING/10_ENGINEERING_CONSTITUTION.md) — engineering standards
- [`06_FOLDER_STRUCTURE.md`](../../01_PROJECT/06_FOLDER_STRUCTURE.md) — repository layout

---

# Implementation Status Summary

| Component | Architecture | Implementation | Testing | Benchmarking |
|-----------|--------------|----------------|---------|--------------|
| Package (`titan_ai.tensor`) | Documented | Implemented | Partial | — |
| Tensor class (core) | Documented | Implemented | Tested | — |
| Dtype / Device | Documented | Implemented | Tested | — |
| NumPy backend | Documented | Implemented | Tested | — |
| Factory methods | Documented | Implemented | Tested | — |
| Shape operations | Documented | Implemented | Tested | — |
| Indexing / slicing | Documented | Implemented | Tested | — |
| Arithmetic / broadcasting | Documented | Implemented | Tested | — |
| Matrix multiplication | Documented | Implemented (1D/2D) | Tested | — |
| Reductions / math ops | Documented | Not started | Not started | Not started |
| Error types (core) | Documented | Implemented | Tested | — |

This table must be updated as implementation progresses. Do not mark items complete until the project's completion criteria are met.
