"""The host half of `primitives.hash`: splitmix64, the pair key and the capacity of a table, which
import without the CUDA stack."""

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
