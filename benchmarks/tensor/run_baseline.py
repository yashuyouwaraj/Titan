"""Lightweight CPU Tensor vs NumPy timing baseline.

This script measures Python/Tensor dispatch overhead against equivalent
NumPy operations. It is not a claim that Titan should beat NumPy.

Usage (from the repository root):

    py benchmarks/tensor/run_baseline.py
"""

from __future__ import annotations

import timeit
from collections.abc import Callable

import numpy as np

from titan_ai.tensor import Dtype, Tensor

REPEAT = 7
NUMBER_SMALL = 80
NUMBER_MEDIUM = 25


def _seconds_per_call(fn: Callable[[], object], *, number: int) -> float:
    fn()
    samples = timeit.repeat(fn, repeat=REPEAT, number=number)
    return min(samples) / number


def _fmt(seconds: float) -> str:
    microseconds = seconds * 1_000_000
    if microseconds >= 1000:
        return f"{microseconds / 1000:.3f} ms"
    return f"{microseconds:.3f} us"


def _ratio(titan: float, numpy_time: float) -> str:
    if numpy_time <= 0:
        return "n/a"
    return f"{titan / numpy_time:.2f}x"


def _pair_for_operation(
    operation: str,
    shape: tuple[int, ...],
    left: Tensor,
    right: Tensor,
    left_np: np.ndarray,
    right_np: np.ndarray,
) -> tuple[Callable[[], object], Callable[[], object]]:
    dtype = Dtype.float64
    np_dtype = np.float64

    if operation == "construction":

        def titan_fn() -> object:
            return Tensor(left_np, dtype=dtype, copy=True)

        def numpy_fn() -> object:
            return np.array(left_np, dtype=np_dtype, copy=True)

        return titan_fn, numpy_fn

    if operation == "zeros":

        def titan_fn() -> object:
            return Tensor.zeros(shape, dtype=dtype)

        def numpy_fn() -> object:
            return np.zeros(shape, dtype=np_dtype)

        return titan_fn, numpy_fn

    if operation == "ones":

        def titan_fn() -> object:
            return Tensor.ones(shape, dtype=dtype)

        def numpy_fn() -> object:
            return np.ones(shape, dtype=np_dtype)

        return titan_fn, numpy_fn

    if operation == "reshape":
        target = (shape[0] * shape[1],)

        def titan_fn() -> object:
            return left.reshape(target)

        def numpy_fn() -> object:
            return left_np.reshape(target)

        return titan_fn, numpy_fn

    if operation == "indexing":

        def titan_fn() -> object:
            return left[:64, :64]

        def numpy_fn() -> object:
            return left_np[:64, :64]

        return titan_fn, numpy_fn

    if operation in {"add", "broadcast_add"}:

        def titan_fn() -> object:
            return left + right

        def numpy_fn() -> object:
            return left_np + right_np

        return titan_fn, numpy_fn

    if operation == "mul":

        def titan_fn() -> object:
            return left * right

        def numpy_fn() -> object:
            return left_np * right_np

        return titan_fn, numpy_fn

    if operation == "matmul":

        def titan_fn() -> object:
            return left @ right

        def numpy_fn() -> object:
            return left_np @ right_np

        return titan_fn, numpy_fn

    if operation == "sum":

        def titan_fn() -> object:
            return left.sum()

        def numpy_fn() -> object:
            return left_np.sum()

        return titan_fn, numpy_fn

    if operation == "mean":

        def titan_fn() -> object:
            return left.mean()

        def numpy_fn() -> object:
            return left_np.mean()

        return titan_fn, numpy_fn

    if operation == "sqrt":

        def titan_fn() -> object:
            return left.sqrt()

        def numpy_fn() -> object:
            return np.sqrt(left_np)

        return titan_fn, numpy_fn

    raise ValueError(operation)


def main() -> None:
    rng = np.random.default_rng(0)
    cases: list[tuple[str, tuple[int, ...], int]] = [
        ("construction", (128, 128), NUMBER_SMALL),
        ("zeros", (128, 128), NUMBER_SMALL),
        ("ones", (128, 128), NUMBER_SMALL),
        ("reshape", (128, 128), NUMBER_SMALL),
        ("indexing", (128, 128), NUMBER_SMALL),
        ("add", (128, 128), NUMBER_SMALL),
        ("mul", (128, 128), NUMBER_SMALL),
        ("broadcast_add", (128, 1), NUMBER_SMALL),
        ("matmul", (64, 64), NUMBER_SMALL),
        ("sum", (128, 128), NUMBER_SMALL),
        ("mean", (128, 128), NUMBER_SMALL),
        ("sqrt", (128, 128), NUMBER_SMALL),
        ("construction", (1024, 1024), NUMBER_MEDIUM),
        ("zeros", (1024, 1024), NUMBER_MEDIUM),
        ("ones", (1024, 1024), NUMBER_MEDIUM),
        ("reshape", (1024, 1024), NUMBER_MEDIUM),
        ("indexing", (1024, 1024), NUMBER_MEDIUM),
        ("add", (1024, 1024), NUMBER_MEDIUM),
        ("mul", (1024, 1024), NUMBER_MEDIUM),
        ("broadcast_add", (1024, 1), NUMBER_MEDIUM),
        ("matmul", (256, 256), NUMBER_MEDIUM),
        ("sum", (1024, 1024), NUMBER_MEDIUM),
        ("mean", (1024, 1024), NUMBER_MEDIUM),
        ("sqrt", (1024, 1024), NUMBER_MEDIUM),
    ]

    print("Titan Tensor CPU baseline (min of repeats, wall time per call)")
    header = (
        f"{'operation':<16} {'shape':<18} {'dtype':<8} "
        f"{'Titan':>12} {'NumPy':>12} {'Titan/NumPy':>12}"
    )
    print(header)

    for operation, shape, number in cases:
        dtype = Dtype.float64
        np_dtype = np.float64
        left_np = rng.random(shape, dtype=np_dtype)
        if operation == "broadcast_add":
            right_np = rng.random((1, shape[0]), dtype=np_dtype)
        elif operation == "matmul":
            right_np = rng.random((shape[0], shape[-1]), dtype=np_dtype)
        else:
            right_np = rng.random(shape, dtype=np_dtype)

        left = Tensor(left_np, dtype=dtype, copy=True)
        right = Tensor(right_np, dtype=dtype, copy=True)
        titan_fn, numpy_fn = _pair_for_operation(operation, shape, left, right, left_np, right_np)
        titan_time = _seconds_per_call(titan_fn, number=number)
        numpy_time = _seconds_per_call(numpy_fn, number=number)
        shape_label = "x".join(str(dim) for dim in shape)
        if operation == "broadcast_add":
            shape_label = f"{shape_label}+{right_np.shape[0]}x{right_np.shape[1]}"
        print(
            f"{operation:<16} {shape_label:<18} {dtype.name:<8} "
            f"{_fmt(titan_time):>12} {_fmt(numpy_time):>12} "
            f"{_ratio(titan_time, numpy_time):>12}"
        )


if __name__ == "__main__":
    main()
