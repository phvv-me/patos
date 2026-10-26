"""Offsets in order by look-back: an array summed by one tile and by many, and a producer's."""

from functools import cache

import cupy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from patos.cuda.primitives import monoid, offsets
from patos.cuda.runtime import Workspace
from patos.cuda.typed import (
    Kernel,
    Per,
    Vector,
    i32,
    kernel,
    lane,
    thread_in_block,
    warp_in_block,
)

_STREAM = cp.cuda.Stream.null

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

_THREADS = 128
# Around a tile of four rows, a few tiles, and enough for a look-back over several windows and for
# the sixteen rows of a vast array.
_SIZES = [1, 31, 1024, 1025, 4097, 40_000, 300_000]


@pytest.mark.parametrize("kind", [np.int32, np.uint32])
@pytest.mark.parametrize("inclusive", [True, False])
@pytest.mark.parametrize("size", _SIZES)
@settings(deadline=None, max_examples=3)
@given(seed=st.integers(0, 2**32 - 1))
def test_an_array_sums_as_the_host_does_alone_and_in_place(
    *, kind: type[np.integer], inclusive: bool, size: int, seed: int
) -> None:
    """A scan is numpy's cumulative sum, wrapping as the type does; exclusive, it opens with 0."""
    info = np.iinfo(kind)
    host = np.random.default_rng(seed).integers(
        info.min, info.max, size, dtype=kind, endpoint=True
    )
    expected = np.cumsum(host, dtype=kind)
    if not inclusive:
        expected = np.concatenate([[0], expected[:-1]]).astype(kind)
    workspace = Workspace(cp)
    copied, in_place = cp.asarray(host), cp.asarray(host)
    apart = offsets.scan(
        copied, out=cp.empty_like(copied), workspace=workspace, stream=_STREAM, inclusive=inclusive
    )
    within = offsets.scan(
        in_place, out=in_place, workspace=workspace, stream=_STREAM, inclusive=inclusive
    )
    assert apart.get().tolist() == within.get().tolist() == expected.tolist()


@cache
def producing(name: str) -> Kernel:
    """A kernel combining each thread's count in order by monoid `name`.

    It writes each thread's combination into `out`, and the one at the end of the last tile into
    `total`.
    """
    ordered, operator = offsets.over(_THREADS), getattr(monoid, name)

    @kernel(per=Per.BLOCK, threads=_THREADS, strided=True)
    def produce(
        chain: offsets.Chain, counts: Vector[i32], out: Vector[i32], total: Vector[i32]
    ) -> None:
        while True:
            tile = ordered.claim(chain)
            if tile * _THREADS >= counts.size:
                return
            index = tile * _THREADS + thread_in_block()
            count = operator.identity(counts[0])
            if index < counts.size:
                count = counts[index]
            start, end = ordered.place(operator, chain, tile, count)
            if index < counts.size:
                out[index] = start
            if (tile + 1) * _THREADS >= counts.size and thread_in_block() == 0:
                total[0] = end

    return produce


@pytest.mark.parametrize("name", ["add", "minimum", "maximum"])
@pytest.mark.parametrize("size", [1, 127, 128, 129, 10_000, 200_000])
def test_a_producer_places_its_counts_in_order_in_one_pass(name: str, size: int) -> None:
    """Each thread gets what the counts before it combine to, the last tile the grand total."""
    counts = np.random.default_rng(size).integers(-9, 9, size, dtype=np.int32)
    ufunc = {"add": np.add, "minimum": np.minimum, "maximum": np.maximum}[name]
    identity = {"add": 0, "minimum": np.iinfo(np.int32).max, "maximum": np.iinfo(np.int32).min}[
        name
    ]
    inclusive = ufunc.accumulate(counts, dtype=np.int32)
    out, total = cp.zeros(size, cp.int32), cp.zeros(1, cp.int32)
    tiles = -(-size // _THREADS)
    chain = offsets.chain(Workspace(cp), tiles, stream=_STREAM, role="test.offsets")
    producing(name)[tiles](chain, cp.asarray(counts), out, total, stream=_STREAM)
    assert out.get().tolist() == np.concatenate([[identity], inclusive[:-1]]).tolist()
    assert total.get().tolist() == [inclusive[-1]]


@cache
def counting(*, warps: bool) -> Kernel:
    """A kernel entering counts in `starts` through `counted`, a thread or a warp's lane 0 each."""
    ordered = offsets.over(_THREADS)

    @kernel(per=Per.BLOCK, threads=_THREADS, strided=True)
    def count(chain: offsets.Chain, counts: Vector[i32], starts: Vector[i32]) -> None:
        while True:
            tile = ordered.claim(chain)
            index = tile * (_THREADS // 32 if warps else _THREADS) + (
                warp_in_block() if warps else thread_in_block()
            )
            if tile * (_THREADS // 32 if warps else _THREADS) >= counts.size:
                return
            mine = index < counts.size and (lane() == 0 or not warps)
            ordered.counted(
                chain,
                tile,
                index if mine else counts.size,
                counts[index] if mine else 0,
                starts,
            )

    return count


@pytest.mark.parametrize("warps", [False, True])
@pytest.mark.parametrize("size", [1, 8, 9, 256, 257, 5000, 70_000])
def test_counted_enters_every_start_and_the_total_from_the_thread_of_the_last_entry(
    *, warps: bool, size: int
) -> None:
    """A thread an entry, or only a warp's first lane, and the rest past the last with nothing."""
    counts = np.random.default_rng(size).integers(0, 9, size, dtype=np.int32)
    starts = cp.full(size + 1, -1, cp.int32)
    per_tile = _THREADS // 32 if warps else _THREADS
    tiles = -(-size // per_tile)
    chain = offsets.chain(Workspace(cp), tiles, stream=_STREAM, role="test.counted")
    counting(warps=warps)[tiles](chain, cp.asarray(counts), starts, stream=_STREAM)
    assert starts.get().tolist() == np.concatenate([[0], np.cumsum(counts)]).tolist()
