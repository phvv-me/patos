"""The host half of `primitives.hash`: splitmix64 and the pair table's builder and lookup, which
import without the CUDA stack so a host-only process can build and read the tables."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

MASK64 = 0xFFFFFFFFFFFFFFFF
# The key of an empty slot: all ones, which a table must never store as a real key.
EMPTY_KEY = MASK64
# The splitmix64 increment, the golden ratio scaled to 64 bits.
GOLDEN_GAMMA = 0x9E3779B97F4A7C15
# The two multipliers of the splitmix64 finalization.
MIX_ONE = 0xBF58476D1CE4E5B9
MIX_TWO = 0x94D049BB133111EB


def pair_key(left: int, *, right: int) -> int:
    """Pack two 32-bit values into one lookup key."""
    return ((left << 32) | right) & MASK64


def splitmix(key: int) -> int:
    """Hash a 64-bit key with splitmix64 finalization."""
    value = (key + GOLDEN_GAMMA) & MASK64
    value ^= value >> 30
    value = (value * MIX_ONE) & MASK64
    value ^= value >> 27
    value = (value * MIX_TWO) & MASK64
    return (value ^ (value >> 31)) & MASK64


def table_capacity(entries: int) -> int:
    """Return a power-of-two capacity with at most one-half load."""
    return 1 << max(1, entries * 2 - 1).bit_length()


def build_pair_slots(entries: Sequence[tuple[int, int]]) -> np.ndarray:
    """Build an open-addressed table from packed keys and 64-bit payloads.

    entries: packed key and payload pairs in insertion order.

    Returns a `uint64` array with shape `[2 * capacity]`, keys at even and payloads at odd slots.
    """
    capacity = table_capacity(len(entries))
    slots = np.zeros(capacity * 2, dtype=np.uint64)
    slots[0::2] = EMPTY_KEY
    mask = capacity - 1
    for key, payload in entries:
        slot = splitmix(key) & mask
        while slots[slot * 2] != EMPTY_KEY:
            slot = (slot + 1) & mask
        slots[slot * 2] = np.uint64(key)
        slots[slot * 2 + 1] = np.uint64(payload)
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
