"""The warp-wide operations numba-cuda lacks, all over the full 32-lane mask.

Call them through the module (`warp.sum(x)`), so `sum` shadows nothing at the call site. All 32
lanes of the warp call one together, a lane with nothing to give passing the neutral value. A
reduction is one name over the types of its operand, picked at compile time: an `i32` or `u32`
reduces in one `redux.sync` (sm_80 and newer, which numba-cuda does not expose and libNVVM lowers
only for some targets, so it is PTX), and a 64-bit integer or a float in five butterfly shuffles.
An operand reduces at its own type, so a sum of two `i32`s, which Numba types as an `i64`, takes
the shuffles until it is converted back (`warp.sum(i32(a + b))`). Read unsigned, -1 is the largest
value, so `warp.min(u32(value))` is the lowest non-negative `value`, or -1 when every lane holds
-1. The scan is CUB's: each step's shuffle also reports whether the source lane exists, so the add
needs no lane test.
"""

import builtins
from typing import NamedTuple

from ..typed import Vector, cuda, device, dispatched, f32, f64, i32, i64, lane, ptx, u8, u32, u64
from . import integers

# The lane mask naming all 32 lanes of a warp.
_FULL_WARP = 0xFFFFFFFF
# One step per offset: shuffle the running total up, and add it where a lane that far down exists.
_SCAN_STEPS = "".join(
    f"shfl.sync.up.b32 received|exists, total, {offset}, 0, {_FULL_WARP:#x};\n"
    "@exists add.s32 total, total, received;\n"
    for offset in (1, 2, 4, 8, 16)
)


@device
def ballot(flag: bool) -> u32:
    """The lanes that set `flag`, lane `i` as bit `i`."""
    return cuda.ballot_sync(_FULL_WARP, flag)


@device
def rank(mask: u32) -> i32:
    """How many lanes `mask` sets below this one."""
    return cuda.popc(u32(mask & cuda.lanemask_lt()))


@device
def first(mask: u32) -> i32:
    """The lowest lane set in `mask`, or -1 when none is."""
    return cuda.ffs(mask) - 1


@device
def _broadcast_signed[T: (i32, i64, f32, f64)](value: T, source: i32) -> T:
    """Numba's shuffle, which takes these four types as they are."""
    return cuda.shfl_sync(_FULL_WARP, value, source)


@device
def _broadcast_u32(value: u32, source: i32) -> u32:
    """Numba would shuffle a `u32` as the `i64` it widens to, so its bits go as an `i32`."""
    return _broadcast_signed(i32(value), source)


@device
def _broadcast_u64(value: u64, source: i32) -> u64:
    """Numba would shuffle a `u64` as the `f64` it converts to, so its bits go as an `i64`."""
    return _broadcast_signed(i64(value), source)


@dispatched(_broadcast_signed, _broadcast_u32, _broadcast_u64)
def broadcast[T: (i32, u32, i64, u64, f32, f64)](value: T, source: i32) -> T:
    """`value` as lane `source` holds it."""
    raise NotImplementedError


@device
def _swapped_signed[T: (i64, f32, f64)](value: T, offset: i32) -> T:
    return cuda.shfl_xor_sync(_FULL_WARP, value, offset)


@device
def _swapped_u64(value: u64, offset: i32) -> u64:
    return _swapped_signed(i64(value), offset)


@dispatched(_swapped_signed, _swapped_u64)
def _swapped[T: (i64, u64, f32, f64)](value: T, offset: i32) -> T:
    """`value` as the lane whose index is this one's xor `offset` holds it, a butterfly step."""
    raise NotImplementedError


@ptx("redux.sync.add.s32 $result, $value, 0xffffffff;")
def _sum_i32(value: i32) -> i32:
    raise NotImplementedError


@ptx("redux.sync.add.u32 $result, $value, 0xffffffff;")
def _sum_u32(value: u32) -> u32:
    raise NotImplementedError


@device
def _sum_shuffled[T: (i64, u64, f32, f64)](value: T) -> T:
    """Each step adds the value of the lane `offset` over, which holds the other half's sum."""
    for offset in (16, 8, 4, 2, 1):
        value = integers.wrapping_add(value, _swapped(value, offset))
    return value


@dispatched(_sum_i32, _sum_u32, _sum_shuffled)
def sum[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
    """The sum of `value` across the warp, wrapping as `T` does."""
    raise NotImplementedError


@ptx("redux.sync.min.s32 $result, $value, 0xffffffff;")
def _min_i32(value: i32) -> i32:
    raise NotImplementedError


@ptx("redux.sync.min.u32 $result, $value, 0xffffffff;")
def _min_u32(value: u32) -> u32:
    raise NotImplementedError


@device
def _min_shuffled[T: (i64, u64, f32, f64)](value: T) -> T:
    """Each step keeps the lower of this lane's value and the one `offset` lanes over."""
    for offset in (16, 8, 4, 2, 1):
        value = builtins.min(value, _swapped(value, offset))
    return value


@dispatched(_min_i32, _min_u32, _min_shuffled)
def min[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
    """The lowest `value` any lane holds."""
    raise NotImplementedError


@ptx("redux.sync.max.s32 $result, $value, 0xffffffff;")
def _max_i32(value: i32) -> i32:
    raise NotImplementedError


@ptx("redux.sync.max.u32 $result, $value, 0xffffffff;")
def _max_u32(value: u32) -> u32:
    raise NotImplementedError


@device
def _max_shuffled[T: (i64, u64, f32, f64)](value: T) -> T:
    """Each step keeps the higher of this lane's value and the one `offset` lanes over."""
    for offset in (16, 8, 4, 2, 1):
        value = builtins.max(value, _swapped(value, offset))
    return value


@dispatched(_max_i32, _max_u32, _max_shuffled)
def max[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
    """The highest `value` any lane holds."""
    raise NotImplementedError


@ptx("redux.sync.and.b32 $result, $value, 0xffffffff;")
def _and_u32(value: u32) -> u32:
    raise NotImplementedError


@device
def _and_shuffled(value: u64) -> u64:
    for offset in (16, 8, 4, 2, 1):
        value &= _swapped(value, offset)
    return value


@dispatched(_and_u32, _and_shuffled)
def and_[T: (u32, u64)](value: T) -> T:
    """The bits `value` sets in every lane."""
    raise NotImplementedError


@ptx("redux.sync.or.b32 $result, $value, 0xffffffff;")
def _or_u32(value: u32) -> u32:
    raise NotImplementedError


@device
def _or_shuffled(value: u64) -> u64:
    for offset in (16, 8, 4, 2, 1):
        value |= _swapped(value, offset)
    return value


@dispatched(_or_u32, _or_shuffled)
def or_[T: (u32, u64)](value: T) -> T:
    """The bits `value` sets in any lane."""
    raise NotImplementedError


@ptx("redux.sync.xor.b32 $result, $value, 0xffffffff;")
def _xor_u32(value: u32) -> u32:
    raise NotImplementedError


@device
def _xor_shuffled(value: u64) -> u64:
    for offset in (16, 8, 4, 2, 1):
        value ^= _swapped(value, offset)
    return value


@dispatched(_xor_u32, _xor_shuffled)
def xor[T: (u32, u64)](value: T) -> T:
    """The bits `value` sets in an odd number of lanes."""
    raise NotImplementedError


@ptx(
    "{\n.reg .s32 total, received;\n.reg .pred exists;\nmov.s32 total, $value;\n"
    + _SCAN_STEPS
    + "mov.s32 $result, total;\n}"
)
def inclusive_sum(value: i32) -> i32:
    """The inclusive prefix sum of `value` across the warp."""
    raise NotImplementedError


class Prefix(NamedTuple):
    """How many lower lanes set a flag, and how many lanes set it in all.

    It is what a compaction needs: a lane that set the flag writes `before` slots behind whatever
    was written before, and the warp advances by `total`.
    """

    before: i32
    total: i32


@device
def ballot_prefix(flag: bool) -> Prefix:
    """The prefix of the lanes that set `flag`."""
    found = ballot(flag)
    return Prefix(rank(found), cuda.popc(found))


@device
def copy[T: (u8, i32)](
    source: Vector[T], source_at: u64, target: Vector[T], target_at: u64, count: u64
) -> None:
    """Copy `count` elements of `source` from `source_at` to `target` from `target_at`.

    The warp's lanes stride the two ranges, which must not overlap, together. Each lane loads four
    elements before it stores them, so four reads are in flight at once.
    """
    index: u64 = lane()
    while index + 96 < count:
        one = source[source_at + index]
        two = source[source_at + index + 32]
        three = source[source_at + index + 64]
        four = source[source_at + index + 96]
        target[target_at + index] = one
        target[target_at + index + 32] = two
        target[target_at + index + 64] = three
        target[target_at + index + 96] = four
        index += 128
    while index < count:
        target[target_at + index] = source[source_at + index]
        index += 32


@device
def reserve(counter: Vector[i32], flag: bool) -> i32:
    """The slot each lane that set `flag` claims at the end of `counter[0]`, -1 for the others.

    A lane with nothing to append passes False. The flagged lanes of the warp take consecutive
    slots in lane order with one atomic add, made by the lowest of them and broadcast to the rest.
    """
    found = ballot(flag)
    slot: i32 = -1
    if found != 0:
        leader = first(found)
        base: i32 = 0
        if lane() == leader:
            base = cuda.atomic.add(counter, 0, cuda.popc(found))
        base = broadcast(base, leader)
        if flag:
            slot = base + rank(found)
    return slot
