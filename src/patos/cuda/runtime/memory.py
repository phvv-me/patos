"""Device scratch buffers reused across calls."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import DTypeLike


class DeviceBuffer[V](Protocol):
    """The slice of a device array a `Workspace` relies on, satisfied by both numpy and cupy: its
    shape and dtype, its leading elements as a view of type `V`, and a fill."""

    def __getitem__(self, index: slice, /) -> V: ...

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

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
    """Device scratch buffers reused across calls, grown to the largest request seen, handed out
    as the array module's own arrays."""

    def __init__(self, arrays: ArrayModule[V]) -> None:
        """Hold scratch for one array module.

        arrays: the array module owning device memory, normally `cupy`.
        """
        self.arrays = arrays
        self.buffers: dict[str, V] = {}
        self.retained: list[V] = []
        self.retention_depth = 0

    def __iter__(self) -> Iterator[str]:
        """Iterate the roles currently held."""
        return iter(self.buffers)

    @contextmanager
    def retain_replaced(self) -> Iterator[None]:
        """Keep replaced cross-stream buffers alive until queued kernels finish."""
        self.retention_depth += 1
        try:
            yield
        finally:
            self._end_retention()

    def take(self, role: str, size: int, dtype: DTypeLike) -> V:
        """Return a buffer of at least `size` elements for `role`, as an exact-size view.

        role: a name unique to the buffer's purpose, since two roles alive at once must
            not share storage.
        size: element count required now.
        dtype: element type; a role asked for another one is reallocated.
        """
        held = self.buffers.get(role)
        if held is None or held.shape[0] < size or held.dtype != dtype:
            if held is not None and self.retention_depth:
                self.retained.append(held)
            held = self.arrays.empty(max(size, 1), dtype=dtype)
            self.buffers[role] = held
        return held[:size]

    def zeros(self, role: str, size: int, dtype: DTypeLike) -> V:
        """Return a zeroed buffer of `size` elements for `role`."""
        view = self.take(role, size, dtype)
        view.fill(0)
        return view

    def _end_retention(self) -> None:
        """Leave one retention scope, dropping the retained buffers when the last one closes."""
        self.retention_depth -= 1
        if not self.retention_depth:
            self.retained.clear()
