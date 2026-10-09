"""Loads at a device address, wider than an element and through the read-only data cache, and the
ranges of an array one thread reads, compares and copies.

`address(array, index)` gives where an element lives and the `load*` functions read an aligned
word, or a vector of them, there in one instruction. They are `ld.global.nc`, so the memory must
stay unwritten while the kernel runs: a table or a text, never a buffer the kernel also writes
or one threads claim with an atomic. An address must be a multiple of the bytes loaded, which a
misaligned load faults on, and `window` reads any byte offset from the aligned words around it.
`pack`, `equal` and `copy` read elements one at a time through plain loads.
"""

from llvmlite import ir
from numba import types
from numba.cuda import cgutils
from numba.cuda.extending import intrinsic

from ..typed import Vector, device, i32, ptx, u8, u32, u64
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


@ptx("ld.global.nc.u32 $result, [$at];")
def load32(at: u64) -> u32:
    """The 4 bytes at the 4-aligned address `at`."""
    raise NotImplementedError


@ptx("ld.global.nc.u64 $result, [$at];")
def load64(at: u64) -> u64:
    """The 8 bytes at the 8-aligned address `at`."""
    raise NotImplementedError


@ptx("ld.global.nc.v4.u32 $result, [$at];")
def load_quad(at: u64) -> tuple[u32, u32, u32, u32]:
    """The four words of the 16 bytes at the 16-aligned address `at`, in one load."""
    raise NotImplementedError


@ptx("ld.global.nc.v2.u64 $result, [$at];")
def load_pair(at: u64) -> tuple[u64, u64]:
    """The two words of the 16 bytes at the 16-aligned address `at`, in one load."""
    raise NotImplementedError


@device
def pack(chars: Vector[u8], at: u64, count: u64) -> tuple[u64, u64]:
    """The `count` bytes of `chars` from byte `at`, at most 16, as two little-endian words.

    One byte at a time, reading none past them and zero above them.
    """
    low = u64(0)
    high = u64(0)
    for offset in range(count):
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
    but stay inside the allocation holding it; its last 20 bytes are packed one at a time.
    """
    if at + 20 > chars.size:
        return pack(chars, at, min(16, chars.size - at))
    where = address(chars, at)
    base = where & ~u64(3)
    shift = (where & 3) << 3
    a, b, c = load32(base), load32(base + 4), load32(base + 8)
    d, e = load32(base + 12), load32(base + 16)
    low = bits.join(bits.funnel(a, b, shift), bits.funnel(b, c, shift))
    high = bits.join(bits.funnel(c, d, shift), bits.funnel(d, e, shift))
    return low, high


@device
def equal(left: Vector[u8], left_at: u64, right: Vector[u8], right_at: u64, count: u64) -> bool:
    """Whether the `count` bytes of `left` from `left_at` are those of `right` from `right_at`.

    The bytes are read in order up to the first that differs.
    """
    offset: u64 = 0
    while offset < count and left[left_at + offset] == right[right_at + offset]:
        offset += 1
    return offset == count


@device
def copy[T: (u8, i32)](
    source: Vector[T], source_at: u64, target: Vector[T], target_at: u64, count: u64
) -> None:
    """Copy `count` elements of `source` from `source_at` to `target` from `target_at`.

    The calling thread copies them alone, through plain loads: a source a kernel wrote is no
    read-only memory. The two ranges must not overlap.
    """
    for offset in range(count):
        target[target_at + offset] = source[source_at + offset]
