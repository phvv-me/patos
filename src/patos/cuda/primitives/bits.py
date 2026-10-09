"""Bit operations numba-cuda lacks, and the SIMD operations on packed lanes.

Call them through the module (`bits.permute(low, high, selector)`); each keeps the operand order of
the CUDA intrinsic it mirrors. `cuda.popc`, `clz`, `ffs`, `brev` and `byte_perm` are numba-cuda's
own. A SIMD operation is one name picked at compile time by the lanes its operands are, as CUDA's
overloads and suffixes pick: `absdiff` of `u8x4` lanes is `__vabsdiffu4`, of `i8x4` lanes
`__vabsdiffs4`, of two `u32` words `__usad(a, b, 0)`. A word converts to lanes for free
(`u8x4(word)`), and lanes to a word (`u32(lanes)`).

The video instructions PTX emulates stay unused (`vabsdiff4.s32` 31 SASS instructions, `vabsdiff2`
12 to 17, `vmin4` 19 to 34): a signed byte is biased into an unsigned one, a half differs through
`sad` on each, and `min` and `max` of bytes are the CUDA headers' SWAR form, six instructions
(seven on sm_121). `vabsdiff4` itself is one instruction from sm_86 to sm_90 and three on sm_121.
"""

from typing import overload

from ..typed import device, dispatched, i8x4, i16, i16x2, i32, ptx, u8x4, u16x2, u32

# Packed lanes of any kind, as a dispatched operation's operands or result.
type Lane = u8x4 | i8x4 | u16x2 | i16x2


@ptx("prmt.b32 $result, $low, $high, $selector;", pure=True)
def permute(low: u32, high: u32, selector: u32) -> u32:
    """The four bytes `selector` picks out of the eight of `high:low`.

    Nibble `i` of `selector` fills byte `i` of the result: its low three bits number a source byte
    (0 to 3 in `low`, 4 to 7 in `high`) and its top bit, when set, fills the byte with that
    byte's sign bit instead.
    """
    raise NotImplementedError


@ptx("shf.r.wrap.b32 $result, $low, $high, $shift;", pure=True)
def funnel(low: u32, high: u32, shift: u32) -> u32:
    """The low 32 bits of `high:low` shifted right by `shift` modulo 32."""
    raise NotImplementedError


# A video instruction takes the operand it adds from a register, never an immediate.
@ptx(
    "{\n.reg .u32 zero;\nmov.u32 zero, 0;\nvabsdiff4.u32.u32.u32 $result, $a, $b, zero;\n}",
    pure=True,
)
def _absdiff_u8x4(a: u8x4, b: u8x4) -> u8x4:
    raise NotImplementedError


@ptx("sad.u32 $result, $a, $b, 0;", pure=True)
def _absdiff_u32(a: u32, b: u32) -> u32:
    raise NotImplementedError


@ptx("sad.s32 $result, $a, $b, 0;", pure=True)
def _absdiff_i32(a: i32, b: i32) -> u32:
    raise NotImplementedError


@device
def _absdiff_i8x4(a: i8x4, b: i8x4) -> u8x4:
    """Bytes biased by 0x80 differ as the signed bytes do."""
    return _absdiff_u8x4(u32(a) ^ 0x80808080, u32(b) ^ 0x80808080)


@device
def _absdiff_u16x2(a: u16x2, b: u16x2) -> u16x2:
    low = _absdiff_u32(u32(a) & 0xFFFF, u32(b) & 0xFFFF)
    return permute(low, _absdiff_u32(u32(a) >> 16, u32(b) >> 16), 0x5410)


@device
def _absdiff_i16x2(a: i16x2, b: i16x2) -> u16x2:
    low = _absdiff_i32(i16(u32(a)), i16(u32(b)))
    return permute(low, _absdiff_i32(i32(u32(a)) >> 16, i32(u32(b)) >> 16), 0x5410)


@overload
def absdiff(a: u8x4, b: u8x4) -> u8x4: ...
@overload
def absdiff(a: i8x4, b: i8x4) -> u8x4: ...
@overload
def absdiff(a: u16x2, b: u16x2) -> u16x2: ...
@overload
def absdiff(a: i16x2, b: i16x2) -> u16x2: ...
@overload
def absdiff(a: u32, b: u32) -> u32: ...
@overload
def absdiff(a: i32, b: i32) -> u32: ...
@dispatched(
    _absdiff_u8x4, _absdiff_i8x4, _absdiff_u16x2, _absdiff_i16x2, _absdiff_u32, _absdiff_i32
)
def absdiff(a: Lane | u32 | i32, b: Lane | u32 | i32) -> Lane | u32:
    """The absolute difference of `a` and `b`, lane by lane, read unsigned."""
    raise NotImplementedError


@ptx("vabsdiff4.u32.u32.u32.add $result, $a, $b, $c;", pure=True)
def _sad_u8x4(a: u8x4, b: u8x4, c: u32) -> u32:
    raise NotImplementedError


@ptx("sad.u32 $result, $a, $b, $c;", pure=True)
def _sad_u32(a: u32, b: u32, c: u32) -> u32:
    raise NotImplementedError


@ptx("sad.s32 $result, $a, $b, $c;", pure=True)
def _sad_i32(a: i32, b: i32, c: u32) -> u32:
    raise NotImplementedError


@device
def _sad_i8x4(a: i8x4, b: i8x4, c: u32) -> u32:
    return _sad_u8x4(u32(a) ^ 0x80808080, u32(b) ^ 0x80808080, c)


@overload
def sad(a: u8x4, b: u8x4, c: u32) -> u32: ...
@overload
def sad(a: i8x4, b: i8x4, c: u32) -> u32: ...
@overload
def sad(a: u32, b: u32, c: u32) -> u32: ...
@overload
def sad(a: i32, b: i32, c: u32) -> u32: ...
@dispatched(_sad_u8x4, _sad_i8x4, _sad_u32, _sad_i32)
def sad(a: Lane | u32 | i32, b: Lane | u32 | i32, c: u32) -> u32:
    """`c` plus the sum of the absolute differences of `a`'s and `b`'s lanes."""
    raise NotImplementedError


@ptx("dp4a.u32.u32 $result, $a, $b, $c;", pure=True)
def _dot_u8x4(a: u8x4, b: u8x4, c: u32) -> u32:
    raise NotImplementedError


@ptx("dp4a.s32.s32 $result, $a, $b, $c;", pure=True)
def _dot_i8x4(a: i8x4, b: i8x4, c: i32) -> i32:
    raise NotImplementedError


@ptx("dp2a.lo.u32.u32 $result, $a, $b, $c;", pure=True)
def _dot_u16x2(a: u16x2, b: u8x4, c: u32) -> u32:
    raise NotImplementedError


@ptx("dp2a.lo.s32.s32 $result, $a, $b, $c;", pure=True)
def _dot_i16x2(a: i16x2, b: i8x4, c: i32) -> i32:
    raise NotImplementedError


@overload
def dot(a: u8x4, b: u8x4, c: u32) -> u32: ...
@overload
def dot(a: i8x4, b: i8x4, c: i32) -> i32: ...
@overload
def dot(a: u16x2, b: u8x4, c: u32) -> u32: ...
@overload
def dot(a: i16x2, b: i8x4, c: i32) -> i32: ...
@dispatched(_dot_u8x4, _dot_i8x4, _dot_u16x2, _dot_i16x2)
def dot(a: Lane, b: Lane, c: u32 | i32) -> u32 | i32:
    """`c` plus the products of `a`'s lanes with `b`'s, two halves with `b`'s two low bytes."""
    raise NotImplementedError


@device
def _at_least(a: u8x4, b: u8x4) -> u32:
    """0xFF in each byte where `a`'s is at least `b`'s, else 0.

    It is the top bit of the bytes' rounded mean of `a` and `~b`, spread over the byte.
    """
    flipped: u32 = u32(b) ^ 0xFFFFFFFF
    mean: u32 = (u32(a) | flipped) - (((u32(a) ^ flipped) & 0xFEFEFEFE) >> 1)
    return permute(mean, 0, 0xBA98)


@device
def _min_u8x4(a: u8x4, b: u8x4) -> u8x4:
    return u32(a) ^ ((u32(a) ^ u32(b)) & _at_least(a, b))


@device
def _max_u8x4(a: u8x4, b: u8x4) -> u8x4:
    return u32(b) ^ ((u32(a) ^ u32(b)) & _at_least(a, b))


@dispatched(_min_u8x4)
def min(a: u8x4, b: u8x4) -> u8x4:
    """The lower of `a`'s and `b`'s lanes, lane by lane."""
    raise NotImplementedError


@dispatched(_max_u8x4)
def max(a: u8x4, b: u8x4) -> u8x4:
    """The higher of `a`'s and `b`'s lanes, lane by lane."""
    raise NotImplementedError
