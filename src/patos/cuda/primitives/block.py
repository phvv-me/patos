"""Reductions and scans across the threads of one block, built on the warp's.

`over(threads)` gives the operations of blocks of that size, which is a compile-time constant so
that the loops over a block's warps unroll and the scratch holds one slot per warp. A kernel
launched with another size refuses to compile them. Every thread of the block calls them
together. Each stages one value per warp in shared memory of its own, which a kernel that calls
it holds once, and leaves the scratch free for the next call.
"""

import builtins
import operator
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

from ..typed import cuda, device, i32, lane, warp_in_block
from . import scalar, warp

_WARP = 32
_MAX_THREADS = 1024


class Block(NamedTuple):
    """The operations across the threads of a block of one size, every thread calling together."""

    sum: Callable[[i32], i32]
    min: Callable[[i32], i32]
    max: Callable[[i32], i32]
    min_nonnegative: Callable[[i32], i32]
    exclusive_sum: Callable[[i32], tuple[i32, i32]]


@cache
def over(threads: int) -> Block:
    """The operations across the threads of a block of `threads` threads.

    threads: the `threads` of every kernel that calls them, a multiple of 32 up to 1024.
    """
    if threads % _WARP or not _WARP <= threads <= _MAX_THREADS:
        raise ValueError(f"a block of {threads} threads is not a multiple of 32 up to 1024")
    return Block(
        sum=_reduction(threads, warp.sum, operator.add, "The sum of `value` across the block."),
        min=_reduction(threads, warp.min, builtins.min, "The lowest `value` any thread holds."),
        max=_reduction(threads, warp.max, builtins.max, "The highest `value` any thread holds."),
        min_nonnegative=_reduction(
            threads,
            warp.min_nonnegative,
            scalar.min_nonnegative,
            "The lowest non-negative `value` any thread holds, or -1 when every thread holds -1.",
        ),
        exclusive_sum=_exclusive_sum(threads),
    )


def _reduction(
    threads: int, across_warp: Callable[[i32], i32], fold: Callable[..., i32], doc: str
) -> Callable[[i32], i32]:
    """The block reduction that reduces each warp with `across_warp` and folds the results.

    across_warp: reduces `value` across a warp.
    fold: combines two of the values `across_warp` gives.
    doc: what it computes, every thread receiving it.
    """
    warps = threads // _WARP

    def reduction(value: i32) -> i32:
        partial = cuda.shared.array(warps, i32)
        reduced_warp = across_warp(value)
        if lane() == 0:
            partial[warp_in_block()] = reduced_warp
        cuda.syncthreads()
        reduced: i32 = partial[0]
        for index in range(1, warps):
            reduced = fold(reduced, partial[index])
        cuda.syncthreads()
        return reduced

    reduction.__doc__ = doc
    return device(reduction, threads=threads)


def _exclusive_sum(threads: int) -> Callable[[i32], tuple[i32, i32]]:
    """The block scan over `threads` threads."""
    warps = threads // _WARP

    def exclusive_sum(value: i32) -> tuple[i32, i32]:
        """`value`'s exclusive prefix sum across the block and the block's total."""
        totals = cuda.shared.array(warps, i32)
        inclusive = warp.inclusive_sum(value)
        if lane() == 31:
            totals[warp_in_block()] = inclusive
        cuda.syncthreads()
        before: i32 = 0
        total: i32 = 0
        for index in range(warps):
            if index < warp_in_block():
                before += totals[index]
            total += totals[index]
        cuda.syncthreads()
        return before + inclusive - value, total

    return device(exclusive_sum, threads=threads)
