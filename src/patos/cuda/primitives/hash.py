"""The open-addressed pair tables the host builds, the kernels probe, and the hash both share."""

from typing import TYPE_CHECKING

import numpy as np

from ..typed import device, i64, u64

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


def build_filter(values: Iterable[int]) -> np.ndarray:
    """Build the one-hash bitmap that holds every value given, and a few it was never given.

    A miss is certain and a hit is a hint, which is the whole use: a probe into the pair table
    is skipped only when the filter says the key cannot be there.

    Returns a `uint64` array with shape `[FILTER_BITS / 64]` of bitmap words.
    """
    words = np.zeros(_FILTER_BITS // 64, dtype=u64)
    for value in values:
        bit = splitmix(value) & (_FILTER_BITS - 1)
        words[bit >> 6] |= 1 << u64(bit & 63)
    return words


@device
def bitmap_holds(words: u64[int], bit: u64) -> bool:
    """Whether bit `bit` of the bitmap packed into 64-bit `words` is set."""
    return (words[bit >> 6] >> (bit & 63)) & 1 != 0


@device
def filter_holds(words: u64[int], value: u64) -> bool:
    """Whether the filter may hold `value`, which is certain only when this is false."""
    return bitmap_holds(words, device_splitmix(value) & (_FILTER_BITS - 1))


@device
def probe(slots: u64[int], mask: u64, key: u64) -> i64:
    """The payload `key` maps to in a table `build_pair_slots` built, or -1 when it holds none.

    mask: the table's capacity less one.
    """
    slot = device_splitmix(key) & mask
    while slots[slot * 2] != EMPTY_KEY:
        if slots[slot * 2] == key:
            return slots[slot * 2 + 1]
        slot = (slot + 1) & mask
    return -1


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
