"""The open-addressed tables the host builds and the kernels read, and the hash both share.

A `PairTable` maps 64-bit keys to 64-bit payloads, a `Bitmap` holds one flag per bit, and a
`Filter` is a bitmap of hashed values whose miss is certain. Each is a record whose device members
read like the Python they mirror: `table.get(key)`, `bit in bitmap`, `value in filter`.
"""

from typing import TYPE_CHECKING

import numpy as np

from ..typed import Struct, device, i64, u64

_MASK64 = 0xFFFFFFFFFFFFFFFF
# The key of an empty slot: all ones, which a table must never store as a real key.
EMPTY_KEY = _MASK64
# The splitmix64 increment, the golden ratio scaled to 64 bits.
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
# The two multipliers of the splitmix64 finalization.
_MIX_ONE = 0xBF58476D1CE4E5B9
_MIX_TWO = 0x94D049BB133111EB

# Bits in a filter. Sixty four thousand against a few hundred members keeps a stray hit under
# one in a hundred, and the eight kilobytes sit in every SM's first-level cache.
_FILTER_BITS = 1 << 16


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def pair_key(left: int, *, right: int) -> int:
    """Pack two 32-bit values into one lookup key."""
    return ((left << 32) | right) & _MASK64


def splitmix(key: int) -> int:
    """Hash a 64-bit key with splitmix64 finalization."""
    value = (key + _GOLDEN_GAMMA) & _MASK64
    value ^= value >> 30
    value = (value * _MIX_ONE) & _MASK64
    value ^= value >> 27
    value = (value * _MIX_TWO) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


@device
def device_splitmix(key: u64) -> u64:
    """`splitmix` over a `u64` key on the device, bit for bit the host body."""
    value = (key + _GOLDEN_GAMMA) & EMPTY_KEY
    value ^= value >> 30
    value = (value * _MIX_ONE) & EMPTY_KEY
    value ^= value >> 27
    value = (value * _MIX_TWO) & EMPTY_KEY
    return value ^ (value >> 31)


def table_capacity(entries: int) -> int:
    """Return a power-of-two capacity with at most one-half load."""
    return 1 << max(1, entries * 2 - 1).bit_length()


def build_pair_slots(entries: Sequence[tuple[int, int]]) -> np.ndarray:
    """Build an open-addressed table from packed keys and 64-bit payloads.

    entries: packed key and payload pairs in insertion order.

    Returns a `uint64` array with shape `[2 * capacity]`, keys at even and payloads at odd slots.
    """
    capacity = table_capacity(len(entries))
    slots = np.zeros(capacity * 2, dtype=u64)
    slots[0::2] = EMPTY_KEY
    mask = capacity - 1
    for key, payload in entries:
        slot = splitmix(key) & mask
        while slots[slot * 2] != EMPTY_KEY:
            slot = (slot + 1) & mask
        slots[slot * 2] = u64(key)
        slots[slot * 2 + 1] = u64(payload)
    return slots


def find_pair(slots: np.ndarray, left: int, *, right: int) -> int:
    """Return a pair table payload, or negative one when absent.

    `slots` is the `uint64` array with shape `[2 * capacity]` that `build_pair_slots` returns.
    """
    key = pair_key(left, right=right)
    mask = slots.size // 2 - 1
    slot = splitmix(key) & mask
    while slots[slot * 2] != EMPTY_KEY:
        if int(slots[slot * 2]) == key:
            return int(slots[slot * 2 + 1])
        slot = (slot + 1) & mask
    return -1


class PairTable(Struct):
    """A table `build_pair_slots` laid out, read on the device as `table.get(key)`.

    slots: keys at even and payloads at odd slots.
    mask: the capacity less one.
    """

    slots: u64[int]
    mask: u64

    @classmethod
    def build(cls, entries: Sequence[tuple[int, int]]) -> PairTable:
        """The table of `entries`, packed key and payload pairs, uploaded."""
        slots = build_pair_slots(entries)
        return cls(slots, slots.size // 2 - 1)

    @device
    def get(self, key: u64) -> i64:
        """The payload `key` maps to, or -1 when the table holds none."""
        slots = self.slots
        slot = device_splitmix(key) & self.mask
        while slots[slot * 2] != EMPTY_KEY:
            if slots[slot * 2] == key:
                return slots[slot * 2 + 1]
            slot = (slot + 1) & self.mask
        return -1


class Bitmap(Struct):
    """A flag per bit packed sixty-four to a word, asked `bit in bitmap`; no bit lies past it."""

    words: u64[int]

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
