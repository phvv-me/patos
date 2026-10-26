"""The passes captured on one device, and what a pass that cannot be captured raises."""

from collections import OrderedDict
from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Self

from cuda.core import Device

from ..backend.errors import CUDAError
from ..runtime.streams import handle
from .arena import Arena
from .recording import Recording

if TYPE_CHECKING:
    from cuda.core import Stream as CoreStream
    from cuda.core.graph import Graph, GraphBuilder

    from ..runtime import Stream, StreamLike


class Unrecordable(Exception):
    """A pass that cannot be recorded: it reads the device back, or the device refused a node.

    error: what was raised while recording, named in the message and kept as the `__cause__`.
    """

    def __init__(self, error: Exception) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.error = error


class _Refusal:
    """A scope that raises `Unrecordable` for any of `errors` the device raises inside it."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _kind, error, _traceback) -> None:
        if isinstance(error, self.errors):
            raise Unrecordable(error) from error

    def __init__(self, *errors: type[Exception]) -> None:
        self.errors = errors


class Captured[Result]:
    """One recorded pass: the graph to replay and what the pass returned while it was recorded.

    result: the pass's return value, whose arrays are the arena memory every replay rewrites.
    """

    def __init__(self, graph: Graph, result: Result) -> None:
        self.graph = graph
        self.result = result


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
        self.refusals = (CUDAError, arrays.cuda.runtime.CUDARuntimeError, RuntimeError)

    def _core(self, stream: StreamLike) -> CoreStream:
        """The cuda.core stream over `stream`, made once."""
        number = handle(stream)
        held = self.streams.get(number)
        if held is None:
            device = Device()
            device.set_current()
            held = self.streams[number] = device.create_stream(stream)
        return held

    def _record[Result](
        self, builder: GraphBuilder, body: Callable[[], Result], stream: Stream
    ) -> tuple[Result, Graph]:
        """Run `body` into `builder` and complete the graph; a device refusal is `Unrecordable`."""
        recording = Recording(self.arrays, builder)
        allocator = self.arrays.cuda.using_allocator(self.arena.malloc)
        with _Refusal(*self.refusals), stream, allocator, recording:
            result = body()
            graph = builder.end_building().complete()
        return result, graph

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

    def launch(self, captured: Captured, stream: StreamLike) -> None:
        """Queue the recorded pass on `stream`."""
        captured.graph.launch(self._core(stream))
