"""Domain-specific exceptions for the Titan AI Tensor Library."""


class TitanTensorError(Exception):
    """Base exception for all Tensor Library errors."""


class TensorConstructionError(TitanTensorError):
    """Raised when tensor construction fails."""


class TensorValidationError(TitanTensorError):
    """Raised when tensor input or state fails validation."""


class UnsupportedDeviceError(TitanTensorError):
    """Raised when a device is not supported."""


class UnsupportedDtypeError(TitanTensorError):
    """Raised when a dtype is not supported."""
