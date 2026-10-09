"""The warp, block and search primitives, each against a host reference."""

import bisect
import re
from collections.abc import Callable, Sequence
from functools import cache

import cupy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numba.core.errors import TypingError
from numba.np.numpy_support import as_dtype

from patos.cuda.primitives import block, integers, search, warp
from patos.cuda.typed import (
    Kernel,
    Matrix,
    Per,
    Struct,
    Vector,
    device,
    i32,
    i64,
    items,
    kernel,
    number,
    thread_index,
    u8,
    u32,
    u64,
)

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

_NUMBERS = [np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64]
# Each reduction's host reference over the values of one warp or block.
_REFERENCES: dict[str, Callable[[np.ndarray], np.generic]] = {
    "sum": lambda values: values.sum(dtype=values.dtype), "min": np.min, "max": np.max,
    "and_": np.bitwise_and.reduce, "or_": np.bitwise_or.reduce, "xor": np.bitwise_xor.reduce,
}  # fmt: skip
# Each reduction over each type it takes: any number, or an unsigned word for the bitwise ones.
_CASES = [
    ("sum", np.int32), ("sum", np.uint32), ("sum", np.int64), ("sum", np.uint64),
    ("sum", np.float32), ("sum", np.float64), ("min", np.int32), ("min", np.uint32),
    ("min", np.int64), ("min", np.uint64), ("min", np.float32), ("min", np.float64),
    ("max", np.int32), ("max", np.uint32), ("max", np.int64), ("max", np.uint64),
    ("max", np.float32), ("max", np.float64), ("and_", np.uint32), ("and_", np.uint64),
    ("or_", np.uint32), ("or_", np.uint64), ("xor", np.uint32), ("xor", np.uint64),
]  # fmt: skip


def ints(kind: type[np.integer]) -> st.SearchStrategy[int]:
    """Integers of `kind`, its edges and 0, 1 and -1 over-represented."""
    info = np.iinfo(kind)
    edges = [info.min, info.max, 0, 1] + ([-1] if info.min < 0 else [])
    return st.one_of(st.integers(info.min, info.max), st.sampled_from(edges))


def numbers(kind: type[np.generic]) -> st.SearchStrategy[int | float]:
    """Values of `kind`; a float is a whole number, which a block sums exactly in any order."""
    if issubclass(kind, np.integer):
        return ints(kind)
    return st.integers(-(2**16), 2**16).map(float)


def lowest_unsigned(values: np.ndarray) -> np.int32:
    """The lowest of `values` read as unsigned, so -1 and any negative lose to any other.

    values: int32 of shape `[n]`.
    """
    return values.view(np.uint32).min().astype(np.int32)


def claimed_runs(taken: Sequence[int], flags: Sequence[bool]) -> list[list[int]]:
    """The slots the flagged lanes of each warp took, in lane order."""
    return [
        [taken[start + lane] for lane in range(32) if flags[start + lane]]
        for start in range(0, len(flags), 32)
    ]


class Probes(Struct):
    """Ranges of sorted values to search for keys, where the two bounds land and the key is."""

    low: Vector[i64]
    high: Vector[i64]
    keys: Vector[number]
    lower: Vector[i64]
    upper: Vector[i64]
    found: Vector[i64]


@cache
def warp_reducing(name: str) -> Kernel:
    """A kernel giving every thread the reduction `name` of the 32 values of its warp.

    Row 0 of the table holds the values, and row 1 gets what each thread receives.
    """
    reduction = getattr(warp, name)

    @kernel(threads=32)
    def reduce(table: Matrix[number]) -> None:
        for lane_item in items(table.shape[1]):
            table[1, lane_item] = reduction(table[0, lane_item])

    return reduce


@cache
def block_reducing(name: str, threads: int) -> Kernel:
    """A kernel giving every thread the reduction `name` of its block, of two rows in a row.

    Rows 2 and 3 of the table get what each thread receives of rows 0 and 1. The second reduction
    reuses the scratch of the first, which the first leaves free.
    """
    reduction = getattr(block.over(threads), name)

    @kernel(per=Per.BLOCK, threads=threads)
    def reduce(table: Matrix[number]) -> None:
        thread = thread_index()
        table[2, thread] = reduction(table[0, thread])
        table[3, thread] = reduction(table[1, thread])

    return reduce


@cache
def block_scanning(threads: int) -> Kernel:
    """A kernel giving every thread its exclusive prefix across its block and the block's total."""
    across = block.over(threads)

    @kernel(per=Per.BLOCK, threads=threads)
    def scan(values: Vector[i32], held: Matrix[i64]) -> None:
        thread = thread_index()
        before, total = across.exclusive_sum(values[thread])
        held[thread, 0] = before
        held[thread, 1] = total

    return scan


_IN_BLOCK = block.over(128)


@kernel(threads=32)
def warp_lowest(table: Matrix[i32]) -> None:
    for lane_item in items(table.shape[1]):
        table[1, lane_item] = warp.min(u32(table[0, lane_item]))


@kernel(per=Per.BLOCK, threads=128)
def block_lowest(table: Matrix[i32]) -> None:
    table[1, thread_index()] = _IN_BLOCK.min(u32(table[0, thread_index()]))


@kernel(threads=32)
def broadcasting(values: Vector[number], sources: Vector[i32], held: Vector[number]) -> None:
    for lane_item in items(values.size):
        held[lane_item] = warp.broadcast(values[lane_item], sources[lane_item])


@kernel(threads=32)
def voting(flags: Vector[u8], held: Matrix[i64]) -> None:
    for lane_item in items(flags.size):
        found = warp.ballot(flags[lane_item] != 0)
        prefix = warp.ballot_prefix(flags[lane_item] != 0)
        held[lane_item, 0] = found
        held[lane_item, 1] = warp.rank(found)
        held[lane_item, 2] = warp.first(found)
        held[lane_item, 3] = prefix.before
        held[lane_item, 4] = prefix.total


@kernel(threads=32)
def reserving(
    counter: Vector[i32], flags: Vector[u8], slots: Vector[i64], held: Vector[i32]
) -> None:
    thread = thread_index()
    slot = warp.reserve(counter, flags[thread] != 0)
    slots[thread] = slot
    if slot >= 0:
        held[slot] = thread


@kernel
def searching(values: Vector[number], probes: Probes) -> None:
    for item in items(probes.keys.size):
        low, high, key = probes.low[item], probes.high[item], probes.keys[item]
        probes.lower[item] = search.lower_bound(values, search.Range(low, high), key)
        probes.upper[item] = search.upper_bound(values, search.Range(low, high), key)
        probes.found[item] = search.find(values, search.Range(low, high), key)


@kernel(per=Per.WARP)
def warp_copying[T: (u8, i32)](source: Vector[T], target: Vector[T], ranges: Matrix[u64]) -> None:
    for item in items(ranges.shape[0]):
        warp.copy(source, ranges[item, 0], target, ranges[item, 1], ranges[item, 2])


@kernel
def dividing[T: (i32, u32, i64, u64)](table: Matrix[T]) -> None:
    """Row 2 gets row 0 divided by row 1, rounded up."""
    for item in items(table.shape[1]):
        table[2, item] = integers.ceildiv(table[0, item], table[1, item])


@kernel
def dividing_by_32(narrow: Vector[i32], wide: Vector[u64]) -> None:
    for item in items(narrow.size):
        narrow[item] = integers.ceildiv(narrow[item], 32)
        wide[item] = integers.ceildiv(wide[item], 32)


@pytest.mark.parametrize(("name", "kind"), _CASES)
@settings(deadline=None, max_examples=15)
@given(data=st.data())
def test_a_warp_reduction_gives_every_lane_what_the_host_reduces_the_warp_to(
    *, name: str, kind: type[np.generic], data: st.DataObject
) -> None:
    """Each of three warps reduces its own 32 values, at the type they are."""
    values = np.array(data.draw(st.lists(numbers(kind), min_size=96, max_size=96)), dtype=kind)
    table = cp.asarray(np.stack([values, np.zeros_like(values)]))
    warp_reducing(name)[96](table)

    reduced = [_REFERENCES[name](values[i : i + 32]) for i in range(0, 96, 32)]
    assert table[1].get().tolist() == np.repeat(reduced, 32).tolist()


@pytest.mark.parametrize(("name", "kind"), _CASES)
@pytest.mark.parametrize("threads", [32, 128])
@settings(deadline=None, max_examples=10)
@given(data=st.data())
def test_a_block_reduction_gives_every_thread_what_the_host_reduces_the_block_to(
    *, name: str, kind: type[np.generic], threads: int, data: st.DataObject
) -> None:
    """Each block reduces its own values, twice in a row through one scratch.

    A block holds one warp or four.
    """
    blocks = data.draw(st.integers(1, 3))
    count = blocks * threads
    drawn = data.draw(st.lists(numbers(kind), min_size=2 * count, max_size=2 * count))
    values = np.array(drawn, dtype=kind).reshape(2, count)
    table = cp.asarray(np.concatenate([values, np.zeros_like(values)]))
    block_reducing(name, threads)[blocks](table)

    groups = values.reshape(2, blocks, threads)
    wanted = [
        [_REFERENCES[name](group) for group in row for _ in range(threads)] for row in groups
    ]
    assert table[2:].get().tolist() == np.array(wanted, dtype=kind).tolist()


@pytest.mark.parametrize(("launched", "width"), [(warp_lowest, 32), (block_lowest, 128)])
@settings(deadline=None, max_examples=25)
@given(data=st.data())
def test_the_unsigned_minimum_is_the_lowest_non_negative_value_or_minus_one(
    *, launched: Kernel, width: int, data: st.DataObject
) -> None:
    """Read unsigned, -1 and every negative value lose to any non-negative one.

    A warp or a block of -1 alone gives -1.
    """
    mixed = st.lists(st.one_of(st.just(-1), ints(np.int32)), min_size=width, max_size=width)
    groups = data.draw(st.lists(st.one_of(st.just([-1] * width), mixed), min_size=1, max_size=3))
    values = np.array(groups, np.int32).ravel()
    table = cp.asarray(np.stack([values, np.zeros_like(values)]))
    launched[len(groups) if width > 32 else values.size](table)

    wanted = np.repeat([lowest_unsigned(np.array(group, np.int32)) for group in groups], width)
    assert table[1].get().tolist() == wanted.tolist()


@pytest.mark.parametrize(("name", "kind"), _CASES)
def test_a_32_bit_integer_reduction_is_one_redux(*, name: str, kind: type[np.generic]) -> None:
    """An `i32` or `u32` reduces in one REDUX on every target; the other types reduce in none."""
    launched = warp_reducing(name)
    launched[32](cp.zeros((2, 32), kind))
    compiled = next(
        found for found in launched.dispatcher.overloads if as_dtype(found[0].dtype) == kind
    )
    sass = launched.dispatcher.inspect_sass(compiled)

    found = re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?(REDUX[\w.]*)", sass)
    assert len(found) == (1 if kind in {np.int32, np.uint32} else 0)


def test_a_reduction_refuses_a_type_it_does_not_take() -> None:
    """A byte reduces at no type of its own, and a signed word has no bitwise reduction."""

    @kernel(threads=32)
    def bytes_summed(table: Matrix[u8]) -> None:
        table[1, thread_index()] = warp.sum(table[0, thread_index()])

    @kernel(threads=32)
    def signed_bits(table: Matrix[i32]) -> None:
        table[1, thread_index()] = warp.or_(table[0, thread_index()])

    taken = r"sum takes \(i32\); \(u32\); \(i64\); \(u64\); \(f32\); \(f64\), not \(u8\)"
    with pytest.raises(TypingError, match=taken):
        bytes_summed[32](cp.zeros((2, 32), np.uint8))
    with pytest.raises(TypingError, match=r"or_ takes \(u32\); \(u64\), not \(i32\)"):
        signed_bits[32](cp.zeros((2, 32), np.int32))


@pytest.mark.parametrize("kind", _NUMBERS)
@settings(deadline=None, max_examples=15)
@given(data=st.data())
def test_a_broadcast_gives_each_lane_the_value_of_the_lane_it_names(
    *, kind: type[np.generic], data: st.DataObject
) -> None:
    """Each lane names its own source lane, and gets that lane's value unchanged."""
    values = np.array(data.draw(st.lists(numbers(kind), min_size=64, max_size=64)), dtype=kind)
    sources = np.array(data.draw(st.lists(st.integers(0, 31), min_size=64, max_size=64)))
    held = cp.zeros(64, kind)
    broadcasting[64](cp.asarray(values), cp.asarray(sources, np.int32), held)

    wanted = [values[(index // 32) * 32 + source] for index, source in enumerate(sources)]
    assert held.get().tolist() == np.array(wanted, dtype=kind).tolist()


@settings(deadline=None, max_examples=40)
@given(flags=st.lists(st.booleans(), min_size=64, max_size=64))
def test_a_ballot_is_the_mask_of_flagged_lanes_ranked_and_led_by_the_lowest(
    *, flags: Sequence[bool]
) -> None:
    """Every lane gets its warp's mask, how many flagged lanes lie below it, and the lowest.

    The prefix of a ballot is that count beside the warp's.
    """
    held = cp.zeros((64, 5), np.int64)
    voting[64](cp.asarray(np.array(flags, np.uint8)), held)

    wanted = []
    for index in range(64):
        start, lane = index - index % 32, index % 32
        mask = sum(flagged << bit for bit, flagged in enumerate(flags[start : start + 32]))
        lowest, below = (mask & -mask).bit_length() - 1, (mask & ((1 << lane) - 1)).bit_count()
        wanted.append([mask, below, lowest, below, mask.bit_count()])
    assert held.get().tolist() == wanted


@settings(deadline=None, max_examples=40)
@given(flags=st.lists(st.booleans(), min_size=96, max_size=96), base=st.integers(0, 50))
def test_a_warp_reserves_a_run_of_slots_with_one_add_for_its_flagged_lanes(
    *, flags: Sequence[bool], base: int
) -> None:
    """The flagged lanes of a warp take consecutive slots in lane order after the counter.

    The warps' runs come in any order and the counter ends past them all.
    """
    counter = cp.array([base], np.int32)
    slots, written = cp.full(96, -2, np.int64), cp.full(base + 97, -2, np.int32)
    reserving[96](counter, cp.asarray(np.array(flags, np.uint8)), slots, written)

    taken, written_at = slots.get().tolist(), written.get().tolist()
    claimed = claimed_runs(taken, flags)
    assert all(not run or run == list(range(run[0], run[0] + len(run))) for run in claimed)
    assert sorted(slot for run in claimed for slot in run) == list(range(base, base + sum(flags)))
    assert taken.count(-1) == flags.count(False)
    assert all(written_at[slot] == thread for thread, slot in enumerate(taken) if flags[thread])
    assert counter.get().tolist() == [base + sum(flags)]


@pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")
def test_a_kernel_refuses_the_block_operations_made_for_another_size_of_block() -> None:
    """A kernel of 128 threads cannot compile a reduction made for 256, however deep it sits."""
    wide = block.over(256)

    @device
    def wide_total(value: i32) -> i32:
        halved = value >> 1
        return wide.sum(halved)

    @kernel(per=Per.BLOCK, threads=128)
    def directly(values: Vector[i32], held: Vector[i64]) -> None:
        held[thread_index()] = wide.sum(values[thread_index()])

    @kernel(per=Per.BLOCK, threads=128)
    def through_a_function(values: Vector[i32], held: Vector[i64]) -> None:
        held[thread_index()] = wide_total(values[thread_index()])

    for reduce in (directly, through_a_function):
        with pytest.raises(TypeError, match=r"runs 128 threads a block .* made for 256"):
            reduce[1](cp.zeros(128, np.int32), cp.zeros(128, np.int64))


def test_a_block_has_whole_warps_and_its_operations_are_made_once_per_size() -> None:
    """A block size that is no multiple of 32, or past 1024, is refused."""
    assert block.over(128) is block.over(128)
    for threads in (0, 16, 48, 1056):
        with pytest.raises(ValueError, match="multiple of 32 up to 1024"):
            block.over(threads)


@pytest.mark.parametrize("threads", [32, 128])
@settings(deadline=None, max_examples=15)
@given(data=st.data())
def test_a_block_scan_gives_every_thread_the_sum_of_those_before_it_and_the_blocks_total(
    *, threads: int, data: st.DataObject
) -> None:
    """The prefix of a thread leaves out its own value, and sums wrap around as an `i32` does."""
    blocks = data.draw(st.integers(1, 3))
    drawn = data.draw(
        st.lists(ints(np.int32), min_size=blocks * threads, max_size=blocks * threads)
    )
    values = np.array(drawn, dtype=np.int32)
    held = cp.zeros((blocks * threads, 2), np.int64)
    block_scanning(threads)[blocks](cp.asarray(values), held)

    segments = values.reshape(blocks, threads)
    before = np.cumsum(segments, axis=1, dtype=np.int32) - segments
    total = np.repeat(segments.sum(axis=1, dtype=np.int32), threads)
    assert held.get().tolist() == np.stack([before.ravel(), total], axis=1).tolist()


@pytest.mark.parametrize(
    ("values_kind", "keys_kind"),
    [(np.int32, np.int32), (np.uint64, np.uint64), (np.int32, np.int64), (np.uint16, np.int64)],
)
@settings(deadline=None, max_examples=25)
@given(data=st.data())
def test_a_bound_is_where_the_host_bisects_a_range_of_the_sorted_values(
    *, values_kind: type[np.integer], keys_kind: type[np.integer], data: st.DataObject
) -> None:
    """A lower bound is the first value not below the key, an upper bound the first above it.

    Each looks in its range alone and gives the end of the range when there is none.
    """
    values = sorted(data.draw(st.lists(ints(values_kind), min_size=1, max_size=40)))
    ends = st.integers(0, len(values))
    drawn = data.draw(st.lists(st.tuples(ends, ends, ints(keys_kind)), min_size=1, max_size=20))
    ranges = [(min(a, b), max(a, b), key) for a, b, key in drawn]
    probes = Probes(
        np.array([low for low, _, _ in ranges], np.int64),
        np.array([high for _, high, _ in ranges], np.int64),
        np.array([key for _, _, key in ranges], keys_kind),
        cp.zeros(len(ranges), np.int64),
        cp.zeros(len(ranges), np.int64),
        cp.zeros(len(ranges), np.int64),
    )
    searching[len(ranges)](cp.asarray(np.array(values, values_kind)), probes)

    lower = [bisect.bisect_left(values, k, a, b) for a, b, k in ranges]
    assert probes.lower.get().tolist() == lower
    assert probes.upper.get().tolist() == [
        bisect.bisect_right(values, k, a, b) for a, b, k in ranges
    ]
    assert probes.found.get().tolist() == [
        at if at < b and values[at] == k else -1
        for at, (_, b, k) in zip(lower, ranges, strict=True)
    ]


@pytest.mark.parametrize("kind", [np.uint8, np.int32])
@settings(deadline=None, max_examples=15)
@given(data=st.data())
def test_a_warp_copies_each_range_with_its_lanes_and_leaves_the_rest(
    *, kind: type[np.integer], data: st.DataObject
) -> None:
    """Ranges up to 600 elements take the four-deep loop and the tail; a copy stays in its slot."""
    ranges_drawn = st.tuples(st.integers(0, 4096), st.integers(0, 600))
    drawn = data.draw(st.lists(ranges_drawn, min_size=1, max_size=4))
    ranges = np.array([[start, 640 * slot, length] for slot, (start, length) in enumerate(drawn)])
    source, target = np.arange(4096 + 600).astype(kind), np.full(640 * len(drawn), 7, kind)
    held = cp.asarray(target)
    warp_copying[len(drawn)](cp.asarray(source), held, cp.asarray(ranges, np.uint64))

    for start, to, length in ranges:
        target[to : to + length] = source[start : start + length]
    assert held.get().tolist() == target.tolist()


@pytest.mark.parametrize("kind", [np.int32, np.uint32, np.int64, np.uint64])
@settings(deadline=None, max_examples=25)
@given(data=st.data())
def test_ceildiv_rounds_the_quotient_up_without_overflowing_near_the_top(
    *, kind: type[np.integer], data: st.DataObject
) -> None:
    """Any value of `T`, its edges included, over any positive divisor of `T`."""
    values = data.draw(st.lists(ints(kind), min_size=64, max_size=64))
    divisors = data.draw(st.lists(st.integers(1, np.iinfo(kind).max), min_size=64, max_size=64))
    table = cp.asarray(np.array([values, divisors, values], kind))
    dividing[64](table)

    wanted = [-(-value // divisor) for value, divisor in zip(values, divisors, strict=True)]
    assert table[2].get().tolist() == wanted


@settings(deadline=None, max_examples=25)
@given(
    narrow=st.lists(ints(np.int32), min_size=64, max_size=64),
    wide=st.lists(ints(np.uint64), min_size=64, max_size=64),
)
def test_ceildiv_by_a_literal_divides_at_the_type_of_the_value(
    *, narrow: Sequence[int], wide: Sequence[int]
) -> None:
    """An int literal takes the type of the value it meets, so no wider signature ties."""
    held = cp.asarray(np.array(narrow, np.int32)), cp.asarray(np.array(wide, np.uint64))
    dividing_by_32[64](*held)

    assert [part.get().tolist() for part in held] == [
        [-(-value // 32) for value in values] for values in (narrow, wide)
    ]


def nsw_count(launched: Kernel, kind: type[np.generic], operation: str) -> int:
    """How many `operation`s the LLVM IR of `launched`, compiled for `kind`, marks `nsw`.

    Numba marks a signed add or multiply as never overflowing, which makes overflow undefined.
    """
    compiled = next(
        found for found in launched.dispatcher.overloads if as_dtype(found[0].dtype) == kind
    )
    return len(re.findall(rf"= {operation} nsw", launched.dispatcher.inspect_llvm(compiled)))


def wrapped(total: int) -> int:
    """`total` as a signed 64-bit integer holds it."""
    return (total + 2**63) % 2**64 - 2**63


def summed_at_the_top(scope: str, kind: type[np.integer]) -> tuple[Kernel, list[int]]:
    """The sum kernel of a warp or of a block of 64, and what it gave its first thread.

    The warp sums 32 values of the top of `kind`. The block sums two rows of 64, the second one
    below the top.
    """
    top = int(np.iinfo(kind).max)
    if scope == "warp":
        launched, table, rows = warp_reducing("sum"), [[top] * 32, [0] * 32], [1]
    else:
        launched = block_reducing("sum", 64)
        table, rows = [[top] * 64, [top - 1] * 64, [0] * 64, [0] * 64], [2, 3]
    held = cp.asarray(np.array(table, kind))
    launched[32 if scope == "warp" else 1](held)
    return launched, [int(held[row, 0]) for row in rows]


@pytest.mark.parametrize("scope", ["warp", "block"])
def test_a_signed_64_bit_sum_wraps_and_marks_no_add_never_to_overflow(scope: str) -> None:
    """Values at the top of `i64` sum to what the integer wraps to, with no `add nsw`.

    Numba marks a signed add as never overflowing, which leaves the wrapped sum to the optimizer;
    the sum adds through `u64`, whose adds it marks nothing, which is the control.
    """
    launched, got = summed_at_the_top(scope, np.int64)
    summed_at_the_top(scope, np.uint64)

    top, width = 2**63 - 1, 32 if scope == "warp" else 64
    assert got == [wrapped(top * width), wrapped((top - 1) * width)][: len(got)]
    assert nsw_count(launched, np.int64, "add") == nsw_count(launched, np.uint64, "add")


def test_the_sum_of_two_i32_operands_is_an_i64_until_it_is_converted_back() -> None:
    """Numba types `a + b` of two `i32`s as an `i64`, so `warp.sum(a + b)` sums in 64 bits.

    That is documented, and `warp.sum(i32(a + b))` is the sum in 32.
    """

    @kernel(threads=32)
    def widening(left: Vector[i32], right: Vector[i32], held: Vector[i64]) -> None:
        for lane_item in items(left.size):
            held[0] = warp.sum(left[lane_item] + right[lane_item])
            held[1] = warp.sum(i32(left[lane_item] + right[lane_item]))

    held = cp.zeros(2, np.int64)
    widening[32](cp.full(32, 2**31 - 1, np.int32), cp.ones(32, np.int32), held)

    assert held.get().tolist() == [2**36, 0]


@pytest.mark.parametrize("kind", [np.int32, np.uint32, np.int64, np.uint64])
def test_ceildiv_rounds_up_at_the_edges_of_every_type(*, kind: type[np.integer]) -> None:
    """Each edge value over each edge divisor, `i64` min included, against Python's integers."""
    info = np.iinfo(kind)
    values = [info.min, info.min + 1, -1, 0, 1, info.max - 1, info.max]
    pairs = [(v, d) for v in values for d in (1, 2, 3, 7, info.max) if v >= info.min]
    columns = [[v for v, _ in pairs], [d for _, d in pairs], [0] * len(pairs)]
    table = cp.asarray(np.array(columns, kind))
    dividing[len(pairs)](table)

    assert table[2].get().tolist() == [-(-v // d) for v, d in pairs]


def test_the_signed_64_bit_ceildiv_multiplies_nothing_that_may_overflow() -> None:
    """`quotient * divisor` leaves `i64` for the minimum, where Numba's `mul nsw` is undefined.

    The remainder decides there, so the signed kernel marks no more multiplies than the unsigned
    one, whose product never passes its value.
    """
    table = cp.asarray(np.array([[0], [1], [0]], np.int64))
    dividing[1](table)
    dividing[1](table.astype(np.uint64))

    assert nsw_count(dividing, np.int64, "mul") == nsw_count(dividing, np.uint64, "mul")
