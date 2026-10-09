"""The open-addressed tables the host builds and the kernels read, and the hash both share.

A `PairTable` maps 64-bit keys to 64-bit payloads, a `Bitmap` holds one flag per bit, and a
`Filter` is a bitmap of hashed values whose miss is certain. Each is a record whose device members
read like the Python they mirror: `table.get(key)`, `bit in bitmap`, `value in filter`.
"""

from typing import TYPE_CHECKING

import numpy as np

from ..hashing import EMPTY_KEY, GOLDEN_GAMMA, MIX_ONE, MIX_TWO, build_pair_slots, splitmix
from ..typed import Struct, Vector, device, i64, u64
from .memory import address, load_pair

# Bits in a filter. Sixty four thousand against a few hundred members keeps a stray hit under
# one in a hundred, and the eight kilobytes sit in every SM's first-level cache.
_FILTER_BITS = 1 << 16


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@device
def device_splitmix(key: u64) -> u64:
    """`splitmix` over a `u64` key on the device, bit for bit the host body."""
    value = (key + GOLDEN_GAMMA) & EMPTY_KEY
    value ^= value >> 30
    value = (value * MIX_ONE) & EMPTY_KEY
    value ^= value >> 27
    value = (value * MIX_TWO) & EMPTY_KEY
    return value ^ (value >> 31)


class PairTable(Struct):
    """A table `build_pair_slots` laid out, read on the device as `table.get(key)`.

    slots: keys at even and payloads at odd slots, a pair to a 16-byte load, so it starts on a
        16-byte boundary as every CuPy allocation does. Nothing writes it while a kernel reads.
    mask: the capacity less one.
    """

    slots: Vector[u64]
    mask: u64

    @classmethod
    def build(cls, entries: Sequence[tuple[int, int]]) -> PairTable:
        """The table of `entries`, packed key and payload pairs, uploaded."""
        slots = build_pair_slots(entries)
        return cls(slots, slots.size // 2 - 1)

    @device
    def get(self, key: u64) -> i64:
        """The payload `key` maps to, or -1 when the table holds none."""
        slot = device_splitmix(key) & self.mask
        while True:
            held, payload = load_pair(address(self.slots, slot * 2))
            if held == EMPTY_KEY:
                return -1
            if held == key:
                return payload
            slot = (slot + 1) & self.mask


class Bitmap(Struct):
    """A flag per bit packed sixty-four to a word, asked `bit in bitmap`; no bit lies past it."""

    words: Vector[u64]

    @device
    def __contains__(self, bit: u64) -> bool:
        return (self.words[bit >> 6] >> (bit & 63)) & 1 != 0

    @classmethod
    def pack(cls, flags: np.ndarray) -> Bitmap:
        """The bitmap whose bit `i` is `flags[i]`, packed into `uint64` words and uploaded.

        flags: a `bool` array with shape `[n]`, `n` a multiple of 64.
        """
        return cls(np.packbits(flags, bitorder="little").view(u64))


class Filter(Struct):
    """A one-hash bitmap holding every value it was built from, and a few it never was.

    A miss is certain and a hit is a hint, which is the whole use: a probe into a pair table is
    skipped only when the filter says the key cannot be there.
    """

    bits: Bitmap

    @device
    def __contains__(self, value: u64) -> bool:
        return (device_splitmix(value) & (_FILTER_BITS - 1)) in self.bits

    @classmethod
    def build(cls, values: Iterable[int]) -> Filter:
        """The filter of `values`, uploaded."""
        flags = np.zeros(_FILTER_BITS, dtype=np.bool_)
        flags[[splitmix(value) & (_FILTER_BITS - 1) for value in values]] = True
        return cls(Bitmap.pack(flags))
