"""SWAR operations over the bytes of a `u32` or `u64` word, eight or four of them at once.

Call them through the module (`bytewise.equal(word, byte)`). A word reads little-endian, byte 0 in
its low bits. Each operation answers in the type of the word it takes, a mask that flags a byte by
setting its top bit (`0x80`) and leaving the rest of it clear, so `count`, `first` and `last` read
a mask and a mask xors, ands and ors with another. A byte of a UTF-8 text is ASCII below `0x80`,
opens a character unless it is `10xxxxxx` and continues it otherwise.
"""

from numba.cuda import libdevice
from numba.cuda.extending import intrinsic

from ..typed import cuda, device, i32, u8, u32, u64

# Eight bytes of one, which times a byte spreads it, and of 0x7F and 0x80.
_ONES = 0x0101010101010101
_LOW7 = 0x7F * _ONES
_HIGH = 0x80 * _ONES


@intrinsic
def _like(_context, word, value):
    """`value` as an integer of the type of `word`, keeping its low bits."""

    def lowered(context, builder, call, arguments):
        return context.cast(builder, arguments[1], call.args[1], word)

    return word(word, value), lowered


@device
def spread(byte: u8) -> u64:
    """`byte` in each of the eight bytes of a `u64`; a `u32` word's is `u32(spread(byte))`."""
    return u64(byte) * _ONES


@device
def zeros[W: (u32, u64)](word: W) -> W:
    """Flag the bytes of `word` that are zero."""
    low = _like(word, _LOW7)
    return _like(word, ~(((word & low) + low) | word | low))


@device
def equal[W: (u32, u64)](word: W, byte: u8) -> W:
    """Flag the bytes of `word` that equal `byte`."""
    return zeros(_like(word, word ^ spread(byte)))


@device
def within[W: (u32, u64)](word: W, low: u8, high: u8) -> W:
    """Flag the bytes of `word` from `low` to `high` inclusive, both below 0x80.

    Every byte of `word` must be ASCII, since a byte from 0x80 up carries into the byte above it.
    """
    from_low = word + _like(word, spread(0x80 - low))
    past_high = word + _like(word, spread(0x7F - high))
    return _like(word, from_low & ~past_high & _like(word, _HIGH))


@device
def ascii[W: (u32, u64)](word: W) -> W:
    """Flag the bytes of `word` below 0x80."""
    return _like(word, ~word & _like(word, _HIGH))


@device
def continuations[W: (u32, u64)](word: W) -> W:
    """Flag the bytes of `word` that continue a character, 10xxxxxx."""
    return _like(word, word & ~(word << 1) & _like(word, _HIGH))


@device
def openings[W: (u32, u64)](word: W) -> W:
    """Flag the bytes of `word` that open a character, any that is not 10xxxxxx."""
    return _like(word, (~word | (word << 1)) & _like(word, _HIGH))


@device
def count[W: (u32, u64)](mask: W) -> i32:
    """How many bytes `mask` flags."""
    return cuda.popc(mask)


@device
def first[W: (u32, u64)](mask: W) -> i32:
    """The index of the lowest byte `mask` flags, or -1 when it flags none."""
    return (i32(cuda.ffs(mask)) - 1) >> 3


@device
def last[W: (u32, u64)](mask: W) -> i32:
    """The index of the highest byte `mask` flags, or -1 when it flags none."""
    return (i32(cuda.popc(_like(mask, -1))) - 1 - i32(cuda.clz(mask))) >> 3


@device
def byte[W: (u32, u64)](word: W, index: u32) -> u32:
    """Byte `index` of `word`, below its byte count."""
    return libdevice.byte_perm(word, word >> 32, index) & 0xFF
