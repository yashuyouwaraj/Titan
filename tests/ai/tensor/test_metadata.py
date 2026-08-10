"""Tests for Tensor metadata properties."""

import numpy as np
import pytest

from titan_ai.tensor import Device, Dtype, Tensor


class TestTensorMetadata:
    def test_shape_1d(self) -> None:
        tensor = Tensor([10, 20, 30])
        assert tensor.shape == (3,)

    def test_shape_2d(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        assert tensor.shape == (2, 3)

    def test_ndim_scalar(self) -> None:
        assert Tensor(5).ndim == 0

    def test_ndim_vector(self) -> None:
        assert Tensor([1, 2, 3]).ndim == 1

    def test_ndim_matrix(self) -> None:
        assert Tensor([[1, 2], [3, 4]]).ndim == 2

    def test_size_scalar(self) -> None:
        assert Tensor(42).size == 1

    def test_size_matrix(self) -> None:
        assert Tensor([[1, 2, 3], [4, 5, 6]]).size == 6

    def test_dtype_default_from_data(self) -> None:
        tensor = Tensor([1, 2, 3])
        assert tensor.dtype == Dtype.int64

    def test_device_defaults_to_cpu(self) -> None:
        tensor = Tensor([1, 2, 3])
        assert tensor.device == Device.cpu()

    def test_metadata_matches_numpy_reference(self) -> None:
        array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        tensor = Tensor(array)
        assert tensor.shape == tuple(array.shape)
        assert tensor.ndim == array.ndim
        assert tensor.size == array.size
        assert tensor.dtype == Dtype.float64

    def test_metadata_properties_are_read_only(self) -> None:
        tensor = Tensor([1, 2, 3])
        with pytest.raises(AttributeError):
            tensor.shape = (5,)  # type: ignore[misc]

    def test_repr(self) -> None:
        tensor = Tensor([1, 2], dtype=Dtype.float32)
        text = repr(tensor)
        assert "Tensor(" in text
        assert "float32" in text
        assert "cpu" in text
