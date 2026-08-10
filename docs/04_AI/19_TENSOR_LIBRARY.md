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

**Currently implemented (Day 2–3):**

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
array = tensor.numpy()        # copy by default
view = tensor.numpy(copy=False)

# Indexing returns Tensor for array-like results, Python scalar for scalars
```

**Planned (later Week 1 days):**

```python
tensor + tensor
tensor.matmul(...)
tensor.sum(...)
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

**Currently implemented (Day 2):**

```text
titan_ai/tensor/
    core/tensor.py          # Tensor class
    backend/base.py         # TensorBackend ABC
    backend/numpy_backend.py
    dtypes/dtypes.py
    devices/devices.py
    exceptions/errors.py
```

The `Tensor` class holds a `TensorBackend` instance and exposes metadata through read-only properties. `NumpyBackend` is the only concrete backend.

**Planned (later days):**

```text
    creation/
    indexing/
    operations/
    reductions/
```

Future GPU backends implement `TensorBackend` without changing the public `Tensor` API.

---

# CPU / NumPy Backend

**Status:** foundation implemented (Day 2).

`NumpyBackend` stores data in a `numpy.ndarray` and implements `TensorBackend` metadata properties. NumPy is used only behind the backend boundary.

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

# Broadcasting Strategy

**Currently implemented:** none.

**Planned semantics (NumPy-compatible):**

1. Compare shapes from trailing dimensions forward
2. Dimensions are compatible when equal, or when one is `1`
3. Scalars broadcast to any shape
4. Incompatible dimensions raise `BroadcastError` with both shapes in the message

Incorrect broadcasting must never silently produce wrong numerical results.

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

**Implemented (Day 2–3):**

| Exception | When |
|-----------|------|
| `TitanTensorError` | Base for all tensor errors |
| `TensorConstructionError` | Invalid or unconvertible input data |
| `TensorValidationError` | Validation failures |
| `UnsupportedDeviceError` | Unsupported device (including CUDA in Week 1) |
| `UnsupportedDtypeError` | Unsupported or unmapped dtype |
| `InvalidShapeError` | Invalid or incompatible shape |
| `InvalidAxisError` | Invalid axis for shape operations |
| `TensorIndexError` | Invalid indexing |

**Planned (later days):** broadcast error, operation-specific errors.

Metadata properties (`shape`, `dtype`, `device`, `ndim`, `size`) are read-only Python properties without setters.

---

# Numerical Correctness Strategy

1. Implement operation against NumPy backend
2. Compare results to direct NumPy computation in tests
3. Use `numpy.testing.assert_allclose` with dtype-appropriate tolerances
4. Cover scalars, vectors, matrices, higher-dimensional tensors, broadcasting, and edge cases
5. No performance claims without benchmarks

---

# Testing Strategy

**Currently implemented (Day 3):** 98 unit tests under `tests/ai/tensor/` covering construction, metadata, dtypes, devices, factories, shape ops, indexing, exceptions, storage ownership, and public API imports.

**Planned coverage (later days):** arithmetic, broadcasting, matmul, reductions, math ops.

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
| Arithmetic / broadcasting | Documented | Not started | Not started | — |
| Reductions / math ops | Documented | Not started | Not started | Not started |
| Error types (core) | Documented | Implemented | Tested | — |

This table must be updated as implementation progresses. Do not mark items complete until the project's completion criteria are met.
