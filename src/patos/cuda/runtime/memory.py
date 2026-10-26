"""Device arrays as the runtime knows them, by the CUDA array interface, and device scratch
buffers reused across calls."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, runtime_checkable

import numpy as np
from cuda.bindings import runtime as cudart

from .streams import checked, handle

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    from numpy.typing import DTypeLike

    from .streams import StreamLike


class ArrayInterface(TypedDict):
    """The CUDA array interface of a device array.

    stream: the handle of the stream that produced the array, which a consumer queuing work on
        another stream waits for; None or absent when no wait is needed.
    """

    typestr: str
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    data: tuple[int, bool]
    stream: NotRequired[int | None]


@runtime_checkable
class Interfaced(Protocol):
    """A device array of any module, known by its CUDA array interface."""

    @property
    def __cuda_array_interface__(self) -> ArrayInterface: ...


class DeviceBuffer[V](Protocol):
    """The slice of an array a `Workspace` relies on, satisfied by both numpy and cupy: its shape,
    dtype and size, its leading elements as a view of type `V`, and a fill."""

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def nbytes(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, index: slice, /) -> V: ...

    def fill(self, value: int, /) -> None:
        """Set every element to `value` on the device."""
        ...


class Allocator[V](Protocol):
    """An array module's `empty`: an uninitialized array of so many elements of a dtype."""

    def __call__(self, shape: int, dtype: DTypeLike) -> V: ...


class ArrayModule[V](Protocol):
    """The allocation a `Workspace` makes, satisfied by both the numpy and cupy modules."""

    @property
    def empty(self) -> Allocator[V]: ...


class Workspace[V: DeviceBuffer]:
    """Device scratch buffers reused across calls, handed out as the array module's own arrays.

    A buffer grows to the capacity class above the largest request seen, a power of two or one and
    a half of one, so a ramp of slowly larger requests reallocates a few times rather than at each
    step, each reallocation dropping the recordings of work that ran in it.

    generation: counts the buffers allocated, so a recording of work that ran in them (see
        `graphs`) knows whether they are still the ones it wrote.
    """

    @contextmanager
    def retain_replaced(self) -> Generator[None]:
        """Keep replaced cross-stream buffers alive until queued kernels finish."""
        self.retention_depth += 1
        try:
            yield
        finally:
            self._end_retention()

    @staticmethod
    def _capacity(size: int) -> int:
        """The least power of two, or one and a half of one, at least `size` and at least one."""
        power = 1 << max(size - 1, 0).bit_length()
        return power * 3 // 4 if power >= 4 and power * 3 // 4 >= size else power

    def __init__(self, arrays: ArrayModule[V]) -> None:
        """Hold scratch for one array module.

        arrays: the array module owning device memory, normally `cupy`.
        """
        self.arrays = arrays
        self.buffers: dict[str, V] = {}
        self.retained: list[V] = []
        self.retention_depth = 0
        self.generation = 0

    def __iter__(self) -> Iterator[str]:
        """Iterate the roles currently held."""
        return iter(self.buffers)

    def _end_retention(self) -> None:
        """Leave one retention scope, dropping the retained buffers when the last one closes."""
        self.retention_depth -= 1
        if not self.retention_depth:
            self.retained.clear()

    def take(self, role: str, size: int, dtype: DTypeLike) -> V:
        """Return a buffer of at least `size` elements for `role`, as an exact-size view.

        A buffer too small or of another dtype is replaced by one of the capacity class of `size`.

        role: a name unique to the buffer's purpose, since two roles alive at once must
            not share storage.
        size: element count required now.
        dtype: element type; a role asked for another one is reallocated.
        """
        held = self.buffers.get(role)
        if held is None or held.shape[0] < size or held.dtype != dtype:
            if held is not None and self.retention_depth:
                self.retained.append(held)
            held = self.arrays.empty(self._capacity(size), dtype=dtype)
            self.buffers[role] = held
            self.generation += 1
        return held[:size]

    def zeros(self, role: str, size: int, dtype: DTypeLike, *, stream: StreamLike) -> V:
        """Return a buffer of `size` elements for `role`, zeroed by work queued on `stream`.

        A host buffer, which no stream reaches, is zeroed at once.
        """
        view = self.take(role, size, dtype)
        nbytes = view.nbytes
        match view:
            case Interfaced():
                address = view.__cuda_array_interface__["data"][0]
                checked(cudart.cudaMemsetAsync(address, 0, nbytes, handle(stream)))
            case _:
                view.fill(0)
        return view
