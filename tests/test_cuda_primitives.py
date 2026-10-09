"""The warp, block, scalar and search primitives, each against a host reference."""

import bisect
from collections.abc import Callable, Sequence
from functools import cache

import cupy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from patos.cuda.primitives import block, scalar, search, warp
from patos.cuda.typed import (
    Kernel,
    Per,
    Struct,
    device,
    i32,
    i64,
    items,
    kernel,
    number,
    thread_index,
    u8,
    u32,
)

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")


def ints(kind: type[np.integer]) -> st.SearchStrategy[int]:
    """Integers of `kind`, its edges and 0, 1 and -1 over-represented."""
    info = np.iinfo(kind)
    edges = [info.min, info.max, 0, 1] + ([-1] if info.min < 0 else [])
    return st.one_of(st.integers(info.min, info.max), st.sampled_from(edges))


def lowest_unsigned(values: np.ndarray) -> np.int32:
    """The lowest of `values` read as unsigned, so -1 and any negative lose to any other.

    values: int32 of shape `[n]`.
    """
    return values.view(np.uint32).min().astype(np.int32)


# Each reduction's host reference over a warp's 32 values, and the type it works in.
_WARP_REDUCTIONS: dict[str, tuple[Callable[[np.ndarray], np.integer], type[np.integer]]] = {
    "sum": (lambda values: values.sum(dtype=np.int32), np.int32),
    "min": (np.min, np.int32),
    "max": (np.max, np.int32),
    "min_unsigned": (np.min, np.uint32),
    "max_unsigned": (np.max, np.uint32),
    "all_bits": (np.bitwise_and.reduce, np.uint32),
    "any_bits": (np.bitwise_or.reduce, np.uint32),
    "odd_bits": (np.bitwise_xor.reduce, np.uint32),
    "min_nonnegative": (lowest_unsigned, np.int32),
}

# Each reduction's host reference over a block's values.
_BLOCK_REDUCTIONS: dict[str, Callable[[np.ndarray], np.integer]] = {
    "sum": _WARP_REDUCTIONS["sum"][0],
    "min": np.min,
    "max": np.max,
    "min_nonnegative": lowest_unsigned,
}


def claimed_runs(taken: Sequence[int], flags: Sequence[bool]) -> list[list[int]]:
    """The slots the flagged lanes of each warp took, in lane order."""
    return [
        [taken[start + lane] for lane in range(32) if flags[start + lane]]
        for start in range(0, len(flags), 32)
    ]


class Probes(Struct):
    """Ranges of sorted values to search for keys, and where the two bounds land."""

    low: i64[int]
    high: i64[int]
    keys: number[int]
    lower: i64[int]
    upper: i64[int]


@cache
def warp_reducing(name: str) -> Kernel:
    """A kernel giving every thread the reduction `name` of the 32 values of its warp."""
    reduction = getattr(warp, name)

    @kernel(threads=32)
    def reduce(values: number[int], held: i64[int]) -> None:
        for lane_item in items(values.size):
            held[lane_item] = reduction(values[lane_item])

    return reduce


@cache
def block_reducing(name: str, threads: int) -> Kernel:
    """A kernel giving every thread the reduction `name` of its block, of two inputs in a row.

    The second reduction reuses the scratch of the first, which the first leaves free.
    """
    reduction = getattr(block.over(threads), name)

    @kernel(per=Per.BLOCK, threads=threads)
    def reduce(values: i32[int], held: i64[int, int]) -> None:
        thread = thread_index()
        held[thread, 0] = reduction(values[thread])
        held[thread, 1] = reduction(values[thread] >> 1)

    return reduce


@cache
def block_scanning(threads: int) -> Kernel:
    """A kernel giving every thread its exclusive prefix across its block and the block's total."""
    across = block.over(threads)

    @kernel(per=Per.BLOCK, threads=threads)
    def scan(values: i32[int], held: i64[int, int]) -> None:
        thread = thread_index()
        before, total = across.exclusive_sum(values[thread])
        held[thread, 0] = before
        held[thread, 1] = total

    return scan


@pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")
def test_a_kernel_refuses_the_block_operations_made_for_another_size_of_block() -> None:
    """A kernel of 128 threads cannot compile a reduction made for 256, however deep it sits."""
    wide = block.over(256)

    @device
    def wide_total(value: i32) -> i32:
        halved = value >> 1
        return wide.sum(halved)

    @kernel(per=Per.BLOCK, threads=128)
    def directly(values: i32[int], held: i64[int]) -> None:
        held[thread_index()] = wide.sum(values[thread_index()])

    @kernel(per=Per.BLOCK, threads=128)
    def through_a_function(values: i32[int], held: i64[int]) -> None:
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


@kernel
def first_lanes(masks: u32[int], held: i64[int]) -> None:
    for item in items(masks.size):
        held[item] = warp.first(masks[item])


@kernel(threads=32)
def reserving(counter: i32[int], flags: u8[int], slots: i64[int], held: i32[int]) -> None:
    thread = thread_index()
    slot = warp.reserve(counter, flags[thread] != 0)
    slots[thread] = slot
    if slot >= 0:
        held[slot] = thread


@kernel
def lowering(best: u32[int], values: i32[int], held: i64[int]) -> None:
    for item in items(held.size):
        held[item] = scalar.min_nonnegative(best[item], values[item])


@kernel
def searching(values: number[int], probes: Probes) -> None:
    for item in items(probes.keys.size):
        low, high, key = probes.low[item], probes.high[item], probes.keys[item]
        probes.lower[item] = search.lower_bound(values, search.Range(low, high), key)
        probes.upper[item] = search.upper_bound(values, search.Range(low, high), key)


@pytest.mark.parametrize("name", _WARP_REDUCTIONS)
@settings(deadline=None, max_examples=25)
@given(data=st.data())
def test_a_warp_reduction_gives_every_lane_what_the_host_reduces_the_warp_to(
    *, name: str, data: st.DataObject
) -> None:
    """Each of three warps reduces its own 32 values."""
    reference, kind = _WARP_REDUCTIONS[name]
    values = np.array(data.draw(st.lists(ints(kind), min_size=96, max_size=96)), dtype=kind)
    held = cp.zeros(96, np.int64)
    warp_reducing(name)[96](cp.asarray(values), held)

    wanted = np.repeat([reference(values[i : i + 32]) for i in range(0, 96, 32)], 32)
    assert held.get().tolist() == wanted.tolist()


@settings(deadline=None, max_examples=40)
@given(masks=st.lists(ints(np.uint32), min_size=1, max_size=40))
def test_the_first_lane_of_a_mask_is_its_lowest_set_bit_or_minus_one(
    *, masks: Sequence[int]
) -> None:
    """A lane is a set bit of the ballot mask, numbered from zero."""
    held = cp.zeros(len(masks), np.int64)
    first_lanes[len(masks)](cp.asarray(np.array(masks, np.uint32)), held)

    assert held.get().tolist() == [(mask & -mask).bit_length() - 1 for mask in masks]


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


@pytest.mark.parametrize("name", _BLOCK_REDUCTIONS)
@pytest.mark.parametrize("threads", [32, 128])
@settings(deadline=None, max_examples=15)
@given(data=st.data())
def test_a_block_reduction_gives_every_thread_what_the_host_reduces_the_block_to(
    *, name: str, threads: int, data: st.DataObject
) -> None:
    """Each block reduces its own values, twice in a row through one scratch.

    A block holds one warp or four.
    """
    blocks = data.draw(st.integers(1, 3))
    drawn = data.draw(
        st.lists(ints(np.int32), min_size=blocks * threads, max_size=blocks * threads)
    )
    values = np.array(drawn, dtype=np.int32)
    held = cp.zeros((blocks * threads, 2), np.int64)
    block_reducing(name, threads)[blocks](cp.asarray(values), held)

    reference = _BLOCK_REDUCTIONS[name]
    wanted = [
        [reference(shifted[start : start + threads]) for shifted in (values, values >> 1)]
        for start in range(0, blocks * threads, threads)
    ]
    assert held.get().tolist() == np.repeat(wanted, threads, axis=0).tolist()


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


@settings(deadline=None, max_examples=40)
@given(pairs=st.lists(st.tuples(ints(np.int32), ints(np.int32)), min_size=1, max_size=40))
def test_the_lower_of_two_values_reads_minus_one_as_the_largest(
    *, pairs: Sequence[tuple[int, int]]
) -> None:
    """Both values read as unsigned, so -1 and every negative lose to any non-negative one."""
    held = cp.zeros(len(pairs), np.int64)
    best, values = (np.array(column, np.int32) for column in zip(*pairs, strict=True))
    lowering[len(pairs)](cp.asarray(best.view(np.uint32)), cp.asarray(values), held)

    wanted = [lowest_unsigned(np.array(pair, np.int32)) for pair in pairs]
    assert held.get().tolist() == wanted


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
    )
    searching[len(ranges)](cp.asarray(np.array(values, values_kind)), probes)

    assert probes.lower.get().tolist() == [
        bisect.bisect_left(values, k, a, b) for a, b, k in ranges
    ]
    assert probes.upper.get().tolist() == [
        bisect.bisect_right(values, k, a, b) for a, b, k in ranges
    ]
