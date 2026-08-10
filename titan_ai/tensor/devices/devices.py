"""Device abstraction for Titan tensors."""

from titan_ai.tensor.exceptions.errors import UnsupportedDeviceError

_SUPPORTED_DEVICES = frozenset({"cpu"})


class Device:
    """Execution device for tensor storage and computation."""

    def __init__(self, device_type: str) -> None:
        normalized = device_type.lower()
        if normalized not in _SUPPORTED_DEVICES:
            if normalized == "cuda":
                raise UnsupportedDeviceError(
                    "CUDA is not supported in Week 1. Only Device('cpu') is available."
                )
            raise UnsupportedDeviceError(
                f"Unsupported device '{device_type}'. Supported devices: cpu."
            )
        self._type = normalized

    @property
    def type(self) -> str:
        """Return the device type identifier (for example, ``'cpu'``)."""
        return self._type

    @classmethod
    def cpu(cls) -> "Device":
        """Return the CPU device."""
        return cls("cpu")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Device):
            return self._type == other._type
        if isinstance(other, str):
            return self._type == other.lower()
        return False

    def __hash__(self) -> int:
        return hash(self._type)

    def __repr__(self) -> str:
        return f"Device('{self._type}')"
