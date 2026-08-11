from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterator

from .errors import NoTTNNDeviceError


class GenerateDeviceSession:
    def __init__(
        self,
        ttnn: Any,
        device_id: int,
        injected: Any | None,
        *,
        trace_region_size: int | None = None,
    ) -> None:
        self.ttnn = ttnn
        self.device_id = device_id
        self.injected = injected
        self.device = None
        self.opened = False
        self.trace_region_size = trace_region_size

    def __enter__(self) -> Any:
        if self.injected is not None:
            self.device = f"fake-device:{self.device_id}"
            return self.device
        open_device = getattr(self.ttnn, "open_device", None)
        if not callable(open_device):
            raise NoTTNNDeviceError("ttnn.open_device is not available")
        kwargs = {"device_id": self.device_id}
        if self.trace_region_size is not None:
            kwargs["trace_region_size"] = int(self.trace_region_size)
        self.device = open_device(**kwargs)
        self.opened = True
        return self.device

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.opened:
            return
        close_device = getattr(self.ttnn, "close_device", None)
        if callable(close_device):
            close_device(self.device)


def maybe_generate_device(
    ttnn: Any,
    device_id: int,
    injected: Any | None,
    *,
    trace_region_size: int | None = None,
) -> GenerateDeviceSession:
    return GenerateDeviceSession(
        ttnn,
        device_id,
        injected,
        trace_region_size=trace_region_size,
    )


@contextmanager
def managed_ttnn_device(
    ttnn: Any,
    device_id: int,
    injected: Any | None = None,
    *,
    trace_region_size: int | None = None,
) -> Iterator[Any]:
    """Own one diagnostic device session and always release it."""
    if injected is not None:
        yield f"fake_device:{device_id}"
        return

    manage_device = (
        None if trace_region_size is not None else getattr(ttnn, "manage_device", None)
    )
    if callable(manage_device):
        try:
            manager = manage_device(device_id=device_id)
        except TypeError:
            manager = manage_device(device_id)
        enter = getattr(manager, "__enter__", None)
        exit_ = getattr(manager, "__exit__", None)
        if not callable(enter) or not callable(exit_):
            raise NoTTNNDeviceError("ttnn manage_device is not a context manager")
        try:
            device = enter()
        except Exception as error:
            raise NoTTNNDeviceError(error) from error
        exc_info = (None, None, None)
        try:
            yield device
        except BaseException:
            exc_info = sys.exc_info()
            raise
        finally:
            exit_(*exc_info)
        return

    open_device = getattr(ttnn, "open_device", None)
    close_device = getattr(ttnn, "close_device", None)
    if callable(open_device):
        try:
            try:
                kwargs = {"device_id": device_id}
                if trace_region_size is not None:
                    kwargs["trace_region_size"] = int(trace_region_size)
                device = open_device(**kwargs)
            except TypeError:
                device = open_device(device_id)
        except Exception as error:
            raise NoTTNNDeviceError(error) from error
        try:
            yield device
        finally:
            if callable(close_device):
                close_device(device)
        return

    create_device = getattr(ttnn, "CreateDevice", None) or getattr(
        ttnn, "create_device", None
    )
    if callable(create_device):
        try:
            device = create_device(device_id)
        except Exception as error:
            raise NoTTNNDeviceError(error) from error
        try:
            yield device
        finally:
            if callable(close_device):
                close_device(device)
        return

    raise NoTTNNDeviceError("ttnn does not expose a device opener")
