"""The passes captured on one device."""

from collections import OrderedDict
from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING

from cuda.core import Device
from cuda.core._utils import cuda_utils

from .arena import Arena
from .captured import Captured
from .recording import Recording
from .refusal import Refusal

if TYPE_CHECKING:
    from cuda.core import Stream as CoreStream
    from cuda.core.graph import Graph, GraphBuilder

    from ..runtime import Stream


class Graphs[Key]:
    """The passes captured on one device, the least recently replayed dropped past a limit."""

    def __init__(self, arrays, *, limit: int = 64) -> None:
        """Start with none captured.

        arrays: the array module owning device memory, normally `cupy`.
        limit: how many recordings to keep; a dropped one is recorded again when asked for.
        """
        self.arrays = arrays
        self.arena = Arena(arrays)
        self.limit = limit
        self.captured: OrderedDict[Key, Captured] = OrderedDict()
        self.streams: dict[int, CoreStream] = {}
        self.refusals = (cuda_utils.CUDAError, arrays.cuda.runtime.CUDARuntimeError, RuntimeError)

    def capture[Result](
        self, key: Key, body: Callable[[], Result], *, stream: Stream
    ) -> Captured[Result]:
        """Record `body` running on `stream` as the pass of `key`, which `body` is not run for.

        `body` runs once with `stream` current and allocates from the arena; it must have run
        before on arrays of the same shapes, so everything it compiles, sizes or caches exists.
        A pass that reads the device back or otherwise cannot be recorded raises `Unrecordable`
        and leaves nothing captured.
        """
        self.arena.rewind()
        core = self._core(stream)
        with ExitStack() as cleanup:
            builder = core.create_graph_builder().begin_building()
            cleanup.callback(builder.close)
            result, graph = self._record(builder, body, stream)
            cleanup.pop_all()
        graph.upload(core)
        captured = Captured(graph, result)
        self.captured[key] = captured
        while len(self.captured) > self.limit:
            self.captured.popitem(last=False)
        return captured

    def drop(self, key: Key) -> None:
        """Forget the recording of `key`, if there is one."""
        self.captured.pop(key, None)

    def get(self, key: Key) -> Captured | None:
        """The recording of `key`, now the most recently used, or None."""
        found = self.captured.get(key)
        if found is not None:
            self.captured.move_to_end(key)
        return found

    def launch(self, captured: Captured, stream: Stream) -> None:
        """Queue the recorded pass on `stream`."""
        captured.graph.launch(self._core(stream))

    def _core(self, stream: Stream) -> CoreStream:
        """The cuda.core stream over the array module's `stream`, made once."""
        held = self.streams.get(stream.ptr)
        if held is None:
            device = Device()
            device.set_current()
            held = self.streams[stream.ptr] = device.create_stream(stream)
        return held

    def _record[Result](
        self, builder: GraphBuilder, body: Callable[[], Result], stream: Stream
    ) -> tuple[Result, Graph]:
        """Run `body` into `builder` and complete the graph; a device refusal is `Unrecordable`."""
        recording = Recording(self.arrays, builder)
        allocator = self.arrays.cuda.using_allocator(self.arena.malloc)
        with Refusal(*self.refusals), stream, allocator, recording:
            result = body()
            graph = builder.end_building().complete()
        return result, graph
