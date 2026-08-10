# Titan AI

> **Understand Everything. Build Everything. Own Everything.**

Titan AI is a first-principles, self-hosted artificial intelligence research and engineering platform. The objective is to design, implement, train, deploy, and continuously improve a complete AI ecosystem without relying on proprietary AI APIs as core dependencies.

## Philosophy

- Understand before implementing.
- Architecture before code.
- Documentation before implementation.
- Local-first and self-hosted operation.
- Modular, replaceable components.
- Measure before optimizing.

## Current Status

| Area | Status |
|------|--------|
| Documentation | Foundation and subsystem architecture documented |
| Implementation | Beginning — Tensor Library foundation (Week 1) |
| Current phase | Day 1 — repository and Python environment setup |

Implementation is starting with the **Tensor Library** (`titan_ai.tensor`), the numerical foundation for autograd, neural networks, transformers, training, and inference.

## Documentation

Engineering documentation lives in [`docs/`](docs/). Start with:

- [`docs/00_FOUNDATION/01_PROJECT_VISION.md`](docs/00_FOUNDATION/01_PROJECT_VISION.md)
- [`docs/00_FOUNDATION/03_MASTER_DOCUMENTATION.md`](docs/00_FOUNDATION/03_MASTER_DOCUMENTATION.md)
- [`docs/04_AI/19_TENSOR_LIBRARY.md`](docs/04_AI/19_TENSOR_LIBRARY.md)

## Requirements

- Python 3.10 or newer

On Windows, the `py` launcher is supported if `python` is not on PATH.

## Environment Setup

Create and activate a virtual environment (recommended):

```bash
py -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the package in editable mode with development dependencies:

```bash
py -m pip install -e ".[dev]"
```

Runtime dependency: **NumPy** (CPU numerical backend for the Tensor Library).

Development dependencies: **pytest**, **ruff**.

PyTorch, CUDA, and GPU libraries are not Week 1 dependencies.

## Verify Installation

```bash
py -c "import titan_ai"
py -c "import titan_ai.tensor"
```

## Run Tests

```bash
py -m pytest
```

The test hierarchy is under `tests/`. Day 1 may have zero tests; pytest should still complete successfully.

## Run Ruff

Lint:

```bash
py -m ruff check titan_ai tests
```

Format check:

```bash
py -m ruff format --check titan_ai tests
```

Apply formatting:

```bash
py -m ruff format titan_ai tests
```

## Project Structure (initial)

```text
titan_ai/          # Python package
tests/             # Test suite (mirrors package layout)
benchmarks/        # Performance baselines
docs/              # Engineering documentation
.ai/               # Project AI configuration
```

The full monorepo layout is defined in [`docs/01_PROJECT/06_FOLDER_STRUCTURE.md`](docs/01_PROJECT/06_FOLDER_STRUCTURE.md) and will be created incrementally as subsystems are implemented.
