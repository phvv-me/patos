"""The scalar types device code spells, and the annotation of an array."""

from typing import TYPE_CHECKING, Protocol, overload

import numpy as np

i16 = np.int16
i32 = np.int32
i64 = np.int64
u8 = np.uint8
u16 = np.uint16
u32 = np.uint32
u64 = np.uint64


class Shaped(Protocol):
    """An array of any module, host or device, known by its shape and its dtype."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...


class Array[T: np.generic]:
    """An array of `T`, as a kernel or device function receives one and as a `Struct` holds one.

    Numba types an array from what the caller passes, so `chars: Array[u8]` converts nothing: it
    fails compilation when `chars` holds anything but `u8`. The members exist for type checkers
    only: indexing and `size` read in a kernel as they do on the device, and the host reads a
    record's array as the numpy or cupy array it holds.
    """

    if TYPE_CHECKING:
        size: int
        shape: tuple[int, ...]
        dtype: np.dtype
        nbytes: int

        @overload
        def __getitem__(self, index: int | np.integer) -> T: ...

        @overload
        def __getitem__(self, index: slice | Shaped) -> Array[T]: ...

        def __getitem__(self, index: int | np.integer | slice | Shaped) -> T | Array[T]: ...

        def __setitem__(
            self, index: int | np.integer | slice | Shaped, value: T | int | Shaped
        ) -> None: ...

        def __add__(self, value: int | Shaped) -> Array[T]: ...

        def __sub__(self, value: int | Shaped) -> Array[T]: ...

        def __mul__(self, value: int) -> Array[T]: ...

        def __neg__(self) -> Array[T]: ...

        def astype(self, dtype: type[np.generic] | np.dtype) -> Array[np.generic]: ...

        def copy(self) -> Array[T]: ...

        def view(self, dtype: type[np.generic] | np.dtype) -> Array[np.generic]: ...
