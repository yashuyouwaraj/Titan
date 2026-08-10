"""Controlled dtype abstraction for Titan tensors."""

from enum import Enum

import numpy as np

from titan_ai.tensor.exceptions.errors import UnsupportedDtypeError

_NUMPY_DTYPE_MAP: dict[np.dtype, "Dtype"] = {}
_NUMPY_TO_TITAN: dict[type, "Dtype"] = {}


class Dtype(Enum):
    """Supported element types for Titan tensors."""

    float32 = "float32"
    float64 = "float64"
    int32 = "int32"
    int64 = "int64"
    bool = "bool"

    def to_numpy(self) -> np.dtype:
        """Convert this Titan dtype to the corresponding NumPy dtype."""
        mapping: dict[Dtype, np.dtype] = {
            Dtype.float32: np.dtype(np.float32),
            Dtype.float64: np.dtype(np.float64),
            Dtype.int32: np.dtype(np.int32),
            Dtype.int64: np.dtype(np.int64),
            Dtype.bool: np.dtype(np.bool_),
        }
        return mapping[self]

    @classmethod
    def from_numpy(cls, np_dtype: np.dtype) -> "Dtype":
        """Convert a NumPy dtype to a Titan dtype."""
        canonical = np.dtype(np_dtype)
        if canonical in _NUMPY_DTYPE_MAP:
            return _NUMPY_DTYPE_MAP[canonical]
        raise UnsupportedDtypeError(
            f"Unsupported dtype {canonical!r}. Supported dtypes: "
            f"{', '.join(member.name for member in cls)}."
        )

    @classmethod
    def infer_from_array(cls, array: np.ndarray) -> "Dtype":
        """Infer a Titan dtype from a NumPy array."""
        return cls.from_numpy(array.dtype)


def _register_numpy_dtypes() -> None:
    for member in Dtype:
        np_dtype = member.to_numpy()
        _NUMPY_DTYPE_MAP[np_dtype] = member
        _NUMPY_TO_TITAN[np_dtype.type] = member


_register_numpy_dtypes()
