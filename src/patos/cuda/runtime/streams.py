"""CUDA streams as the host runtime sees them: protocols for what a pipeline calls on one, and
the wait that lets every stream read freshly uploaded tables."""

from typing import TYPE_CHECKING, Protocol, Self

if TYPE_CHECKING:
    from types import TracebackType


class Event(Protocol):
    """One CUDA event, ordering work across streams."""

    def record(self, stream: Stream | None = None) -> None:
        """Record completion of work already queued on a stream."""
        ...

    def synchronize(self) -> None:
        """Wait on the host until the recorded work finishes."""
        ...


class Stream(Protocol):
    """A CUDA stream as the host runtime uses one: entered as the current stream, waited on, and
    made to wait for another stream's event."""

    def __enter__(self) -> Self:
        """Enter this stream as the current stream."""
        ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the stream context."""
        ...

    @property
    def ptr(self) -> int:
        """The stream's handle, which is 0 for the default stream."""
        ...

    def synchronize(self) -> None:
        """Wait for queued work to finish."""
        ...

    def wait_event(self, _event: Event) -> None:
        """Queue a dependency on a previously recorded event."""
        ...


def settle_uploads(arrays) -> None:
    """Wait for the tables just uploaded on the current stream, so any stream may read them.

    arrays: the array module owning device memory, normally `cupy`; one without streams, such
        as numpy, has nothing to settle.
    """
    try:
        stream = arrays.cuda.get_current_stream()
    except AttributeError:
        return
    stream.synchronize()
