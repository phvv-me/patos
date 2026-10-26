"""CUDA streams as the host runtime sees them: the stream protocol every call that queues work
takes, the stream a capture enters, and the wait that lets every stream read freshly uploaded
tables.

Nothing here reads an ambient "current stream": work is queued on the stream the caller names.
"""

from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING, Protocol, Self

from cuda.bindings import runtime as cudart
from cuda.core import Stream as CoreStream

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from types import TracebackType


class StreamLike(Protocol):
    """A stream of any library, known by the CUDA stream protocol: CuPy's, torch's, cuda.core's."""

    def __cuda_stream__(self) -> tuple[int, int]:
        """The protocol's version and the stream's handle."""
        ...


class Event(Protocol):
    """One CUDA event, ordering work across streams."""

    @property
    def ptr(self) -> int:
        """The event's handle."""
        ...

    def record(self, stream: Stream) -> None:
        """Record completion of work already queued on `stream`."""
        ...

    def synchronize(self) -> None:
        """Wait on the host until the recorded work finishes."""
        ...


class Stream(StreamLike, Protocol):
    """A stream an array module enters as its current one, so its own calls queue there too, as a
    capture needs, and that waits on the host or for another stream's event: CuPy's."""

    @property
    def ptr(self) -> int:
        """The stream's handle, which is 0 for the default stream."""
        ...

    def __enter__(self) -> Self:
        """Enter this stream as the array module's current stream."""
        ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the array module's stream."""
        ...

    def synchronize(self) -> None:
        """Wait on the host for queued work to finish."""
        ...

    def wait_event(self, event: Event, /) -> None:
        """Queue a dependency on a previously recorded event."""
        ...


def handle(stream: StreamLike) -> int:
    """The handle `stream` names by the CUDA stream protocol."""
    return stream.__cuda_stream__()[1]


@cache
def borrowed(number: int) -> CoreStream:
    """The stream of handle `number` as a cuda.core stream, borrowed once."""
    return CoreStream.from_handle(number)


def checked(result: Sequence) -> tuple:
    """`result` of a CUDA runtime call, past its error code, which raises unless it is success."""
    error, *values = result
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA runtime call failed: {error.name}")
    return tuple(values)


def ordered(*, producer: int, consumer: int) -> None:
    """Make the stream of handle `consumer` wait for the work queued so far on `producer`.

    The event lives as long as the call: a wait holds what the event marked when it was queued.
    """
    if producer == consumer:
        return
    with _event() as event:
        checked(cudart.cudaEventRecord(event, producer))
        checked(cudart.cudaStreamWaitEvent(consumer, event, 0))


def settle_uploads(stream: StreamLike) -> None:
    """Wait for the tables just uploaded on `stream`, so any stream may read them."""
    borrowed(handle(stream)).sync()


@contextmanager
def _event() -> Generator[cudart.cudaEvent_t]:
    """An event without timing, destroyed on leaving."""
    (event,) = checked(cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming))
    try:
        yield event
    finally:
        checked(cudart.cudaEventDestroy(event))
