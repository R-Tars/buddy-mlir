from __future__ import annotations


NO_TTNN_DEVICE_MESSAGE = "No TTNN device detected. Use --dry-run or run on P150A."


class NoTTNNDeviceError(RuntimeError):
    pass
