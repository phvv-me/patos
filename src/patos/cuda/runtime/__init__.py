"""The host side of a kernel library: the streams a pipeline orders work on (`streams`), and device
arrays and scratch reused across calls (`memory`). Launching is the kernel's own (`typed.Kernel`).
Every call that queues work takes the stream it queues on."""

from .memory import Allocator, ArrayInterface, ArrayModule, DeviceBuffer, Interfaced, Workspace
from .streams import Event, Stream, StreamLike, settle_uploads

__all__ = [
    "Allocator", "ArrayInterface", "ArrayModule", "DeviceBuffer", "Event", "Interfaced", "Stream",
    "StreamLike", "Workspace", "settle_uploads",
]  # fmt: skip
