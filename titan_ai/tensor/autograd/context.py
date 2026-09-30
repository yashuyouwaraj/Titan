"""Gradient tracking context for autograd system."""

from threading import local

_thread_local = local()


class _TrackingState:
    """Internal tracking state holder."""
    
    def __init__(self) -> None:
        self._enabled: bool = True


def _get_state() -> _TrackingState:
    """Get or create thread-local tracking state."""
    if not hasattr(_thread_local, "state"):
        _thread_local.state = _TrackingState()
    return _thread_local.state


def enable_tracking() -> None:
    """Enable gradient tracking globally."""
    _get_state()._enabled = True


def disable_tracking() -> None:
    """Disable gradient tracking globally."""
    _get_state()._enabled = False


def is_tracking() -> bool:
    """Check if gradient tracking is currently enabled."""
    return _get_state()._enabled
