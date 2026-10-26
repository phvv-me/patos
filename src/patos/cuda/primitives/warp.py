"""The warp-wide operations numba-cuda lacks, all over the full 32-lane mask.

Call them through the module (`warp.sum(x)`), so `sum` shadows nothing at the call site. All 32
lanes of the warp call one together, a lane with nothing to give passing the neutral value. A
reduction is `reduce` at the `Monoid` of its name (`sum` is `reduce(add, value)`), which runs each
built-in monoid and type the fastest exact way (`redux`: one `redux.sync`, or several on the halves
of a 64-bit integer or the key of a float) and any other in five butterfly shuffles. An operand
reduces at its own type, so a sum of two `i32`s, which Numba types as an `i64`, takes the wide path
until it is converted back (`warp.sum(i32(a + b))`). Read unsigned, -1 is the largest value, so
`warp.min(u32(value))` is the lowest non-negative `value`, or -1 when every lane holds -1.
`scan` of an `i32` or `u32` sum is CUB's `inclusive_sum`: each step's shuffle also reports whether
the source lane exists, so the add needs no lane test; `exclusive` scans from it.
"""

from collections.abc import Callable, Sequence

from numba import types

from ..typed import (
    Struct,
    Vector,
    cuda,
    device,
    dispatched,
    f32,
    f64,
    i32,
    i64,
    lane,
    ptx,
    selected,
    u8,
    u32,
    u64,
)
from . import redux
from .monoid import Monoid, add, bitwise_and, bitwise_or, bitwise_xor, maximum, minimum

_WORDS = (types.int32, types.uint32)
_WARP = 32
_BROADCAST, _SWAP, _LOWER = 0, 1, 2

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
def _shuffle_signed[T: (i32, i64, f32, f64)](value: T, how: i32, lane_or_offset: i32) -> T:
    if how == _BROADCAST:
        return cuda.shfl_sync(_FULL_WARP, value, lane_or_offset)
    if how == _SWAP:
        return cuda.shfl_xor_sync(_FULL_WARP, value, lane_or_offset)
    return cuda.shfl_up_sync(_FULL_WARP, value, lane_or_offset)


@device
def _shuffle_u32(value: u32, how: i32, lane_or_offset: i32) -> u32:
    """Numba would shuffle a `u32` as the `i64` it widens to, so its bits go as an `i32`."""
    return _shuffle_signed(i32(value), how, lane_or_offset)


@device
def _shuffle_u64(value: u64, how: i32, lane_or_offset: i32) -> u64:
    """Numba would shuffle a `u64` as the `f64` it converts to, so its bits go as an `i64`."""
    return _shuffle_signed(i64(value), how, lane_or_offset)


@dispatched(_shuffle_signed, _shuffle_u32, _shuffle_u64)
def _shuffle[T: (i32, u32, i64, u64, f32, f64)](value: T, how: i32, lane_or_offset: i32) -> T:
    """`value` as another lane holds it: by lane, by xor of lane, or by the lane below."""
    raise NotImplementedError


@device
def broadcast[T: (i32, u32, i64, u64, f32, f64)](value: T, source: i32) -> T:
    """`value` as lane `source` holds it."""
    return _shuffle(value, _BROADCAST, source)


@device
def _swapped[T: (i32, u32, i64, u64, f32, f64)](value: T, offset: i32) -> T:
    """`value` as the lane whose index is this one's xor `offset` holds it, a butterfly step."""
    return _shuffle(value, _SWAP, offset)


@device
def _lowered[T: (i32, u32, i64, u64, f32, f64)](value: T, offset: i32) -> T:
    """`value` as the lane `offset` below holds it, or as this lane does where none is."""
    return _shuffle(value, _LOWER, offset)


@device
def butterfly[M: Monoid, T: (i32, u32, i64, u64, f32, f64)](monoid: M, value: T) -> T:
    """The `monoid` combination of `value` across the warp in five shuffles, whatever the type.

    Each step combines this lane's value with that of the lane `offset` over, which holds the
    combination of the other half. It is what `reduce` runs where no `redux` strategy is faster.
    """
    for offset in (16, 8, 4, 2, 1):
        value = monoid.combine(value, _swapped(value, offset))
    return value


def _reduction(operands: Sequence[types.Type]) -> Callable:
    """The strategy `redux` names for a monoid and the type of its operand, else the butterfly."""
    return redux.strategy(operands) or butterfly


@selected(_reduction)
def reduce[M: Monoid, T: (i32, u32, i64, u64, f32, f64)](monoid: M, value: T) -> T:
    """The `monoid` combination of `value` across the warp, whose operator commutes.

    The built-in monoids reduce each type by the fastest exact way (`redux`), and any other by
    `butterfly`; the call compiles to one of them.
    """
    raise NotImplementedError


@device
def _shuffled_scan[M: Monoid, T: (i32, u32, i64, u64, f32, f64)](monoid: M, value: T) -> T:
    for offset in (1, 2, 4, 8, 16):
        below = _lowered(value, offset)
        if lane() >= offset:
            value = monoid.combine(below, value)
    return value


@device
def _predicated_sum[M: Monoid, T: (i32, u32)](monoid: M, value: T) -> T:
    return type(value)(inclusive_sum(value))


def _scanning(operands: Sequence[types.Type]) -> Callable:
    """`inclusive_sum` for a 32-bit integer sum, else the shuffles."""
    monoid, value = operands
    summed = redux.named().get(monoid) == "add" and types.unliteral(value) in _WORDS
    return _predicated_sum if summed else _shuffled_scan


@selected(_scanning)
def scan[M: Monoid, T: (i32, u32, i64, u64, f32, f64)](monoid: M, value: T) -> T:
    """The `monoid` combination of `value` across the lanes up to this one, this one included."""
    raise NotImplementedError


@device
def exclusive[M: Monoid, T: (i32, u32, i64, u64, f32, f64)](monoid: M, value: T) -> T:
    """The `monoid` combination of `value` across the lanes below this one, lane 0 the identity."""
    before = _lowered(scan(monoid, value), 1)
    if lane() == 0:
        before = monoid.identity(value)
    return before


@device
def sum[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
    """The sum of `value` across the warp, wrapping as `T` does."""
    return reduce(add, value)


@device
def min[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
    """The lowest `value` any lane holds."""
    return reduce(minimum, value)


@device
def max[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
    """The highest `value` any lane holds."""
    return reduce(maximum, value)


@device
def and_[T: (u32, u64)](value: T) -> T:
    """The bits `value` sets in every lane."""
    return reduce(bitwise_and, value)


@device
def or_[T: (u32, u64)](value: T) -> T:
    """The bits `value` sets in any lane."""
    return reduce(bitwise_or, value)


@device
def xor[T: (u32, u64)](value: T) -> T:
    """The bits `value` sets in an odd number of lanes."""
    return reduce(bitwise_xor, value)


@ptx(
    "{\n.reg .s32 total, received;\n.reg .pred exists;\nmov.s32 total, $value;\n"
    + _SCAN_STEPS
    + "mov.s32 $result, total;\n}"
)
def inclusive_sum(value: i32) -> i32:
    """The inclusive prefix sum of `value` across the warp."""
    raise NotImplementedError


class Prefix(Struct):
    """How many lower lanes set a flag, and how many lanes set it in all.

    It is what a compaction needs: a lane that set the flag writes `before` slots behind whatever
    was written before, and the warp advances by `total`.
    """

    before: i32
    total: i32


@device
def ballot_prefix(flag: bool) -> Prefix:
    """The prefix of the lanes that set `flag`.

    It is the exclusive `add` scan of the flag as a 0 or 1 and its reduction, each in one
    population count of the ballot, which no shuffle matches.
    """
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
def claim(counter: Vector[i32], count: i32) -> i32:
    """The first of the `count` consecutive slots this lane claims at the end of `counter[0]`.

    A lane with nothing to claim passes 0. The warp makes one atomic add of everything it claims,
    and the slots of each lane follow those of the lanes below it, so the order within a warp is
    the lane order and between warps is the order the atomics land in.
    """
    before = exclusive(add, count)
    total = broadcast(before + count, _WARP - 1)
    base: i32 = 0
    if lane() == 0 and total != 0:
        base = cuda.atomic.add(counter, 0, total)
    return broadcast(base, 0) + before


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
