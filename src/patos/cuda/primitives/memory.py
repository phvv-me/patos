"""Loads at a device address, wider than an element and through the read-only data cache.

`address(array, index)` gives where an element lives and the `load*` functions read an aligned
word, or a vector of them, there in one instruction. They are `ld.global.nc`, so the memory must
stay unwritten while the kernel runs: a table or a text, never a buffer the kernel also writes
or one threads claim with an atomic. An address must be a multiple of the bytes loaded, which a
misaligned load faults on, and `window` reads any byte offset from the aligned words around it.
"""

from llvmlite import ir
from numba import types
from numba.cuda import cgutils
from numba.cuda.extending import intrinsic

from ..typed import Vector, device, ptx, u8, u32, u64
from . import bits


@intrinsic
def address(_context, array, index):
    """The device address of `array[index]` as a `u64`, `index` up to one past the end."""

    def lowered(context, builder, _call, arguments):
        held, element = arguments
        record = context.make_array(array)(context, builder, held)
        element = context.cast(builder, element, index, types.intp)
        pointer = cgutils.get_item_pointer(context, builder, array, record, [element])
        return builder.ptrtoint(pointer, ir.IntType(64))

    return types.uint64(array, index), lowered


@ptx("ld.global.nc.u32 $result, [$at];", pure=True)
def load32(at: u64) -> u32:
    """The 4 bytes at the 4-aligned address `at`."""
    raise NotImplementedError


@ptx("ld.global.nc.u64 $result, [$at];", pure=True)
def load64(at: u64) -> u64:
    """The 8 bytes at the 8-aligned address `at`."""
    raise NotImplementedError


@ptx("ld.global.nc.v4.u32 $result, [$at];", pure=True)
def load_quad(at: u64) -> tuple[u32, u32, u32, u32]:
    """The four words of the 16 bytes at the 16-aligned address `at`, in one load."""
    raise NotImplementedError


@ptx("ld.global.nc.v2.u64 $result, [$at];", pure=True)
def load_pair(at: u64) -> tuple[u64, u64]:
    """The two words of the 16 bytes at the 16-aligned address `at`, in one load."""
    raise NotImplementedError


@device
def _tail(chars: Vector[u8], at: u64) -> tuple[u64, u64]:
    """`window` one byte at a time."""
    low = u64(0)
    high = u64(0)
    for offset in range(min(16, chars.size - at)):
        if offset < 8:
            low |= u64(chars[at + offset]) << (8 * offset)
        else:
            high |= u64(chars[at + offset]) << (8 * (offset - 8))
    return low, high


@device
def window(chars: Vector[u8], at: u64) -> tuple[u64, u64]:
    """The 16 bytes of `chars` from byte `at` as two little-endian words, zero past the end.

    at: a byte offset inside `chars`.

    Five aligned 4-byte loads and four funnel shifts, which may read up to 3 bytes before `chars`
    but stay inside the allocation holding it; its last 20 bytes are read one at a time.
    """
    if at + 20 > chars.size:
        return _tail(chars, at)
    where = address(chars, at)
    base = where & ~u64(3)
    shift = (where & 3) << 3
    a, b, c = load32(base), load32(base + 4), load32(base + 8)
    d, e = load32(base + 12), load32(base + 16)
    low = (u64(bits.funnel(b, c, shift)) << 32) | bits.funnel(a, b, shift)
    high = (u64(bits.funnel(d, e, shift)) << 32) | bits.funnel(c, d, shift)
    return low, high
