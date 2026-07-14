from __future__ import annotations

from typing import Any

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
