"""The scalar types device code spells, and the annotation of a device array."""

from typing import TYPE_CHECKING

import numpy as np

i16 = np.int16
i32 = np.int32
i64 = np.int64
u8 = np.uint8
u16 = np.uint16
u32 = np.uint32
u64 = np.uint64


class Array[T: np.generic]:
    """A device array of `T`, as a kernel or device function receives one.

    Numba types an array from what the caller passes, so `chars: Array[u8]` converts nothing: it
    fails compilation when `chars` holds anything but `u8`. The members exist for type checkers
    only, so that indexing and `size` read in a kernel as they do on the device.
    """

    if TYPE_CHECKING:
        size: int
        shape: tuple[int, ...]

        def __getitem__(self, index: int | np.integer) -> T: ...

        def __setitem__(self, index: int | np.integer, value: T) -> None: ...
