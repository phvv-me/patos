"""Reductions and scans across the threads of one block, built on the warp's.

`over(threads)` gives the operations of blocks of that size, which is a compile-time constant so
that the loops over a block's warps unroll and the scratch holds one slot per warp. A kernel
launched with another size refuses to compile them. Every thread of the block calls them
together. A reduction takes the types its warp reduction does (`warp.sum` and the rest), so the
lowest non-negative value is `min(u32(value))` here too. Each stages one value per warp in shared
memory of its own, which a kernel that calls it holds once per type, and leaves the scratch free
for the next call.
"""

import builtins
from collections.abc import Callable
from functools import cache
from typing import NamedTuple, Protocol

from ..typed import Vector, cuda, device, f32, f64, i32, i64, lane, number, u32, u64, warp_in_block
from . import integers, warp

_WARP = 32
_MAX_THREADS = 1024


class Reduction(Protocol):
    """A block reduction of numbers, as a type checker reads it."""

    def __call__[T: (i32, u32, i64, u64, f32, f64)](self, value: T) -> T: ...


class Bitwise(Protocol):
    """A block reduction of bits, as a type checker reads it."""

    def __call__[T: (u32, u64)](self, value: T) -> T: ...


class Block(NamedTuple):
    """The operations across the threads of a block of one size, every thread calling together."""

    sum: Reduction
    min: Reduction
    max: Reduction
    and_: Bitwise
    or_: Bitwise
    xor: Bitwise
    exclusive_sum: Callable[[i32], tuple[i32, i32]]


@device
def _staged[T: (i32, u32, i64, u64, f32, f64)](reduced: T, partial: Vector[number]) -> None:
    """Hold each warp's `reduced` in its slot of `partial`, for every thread of the block."""
    if lane() == 0:
        partial[warp_in_block()] = reduced
    cuda.syncthreads()


@cache
def over(threads: int) -> Block:
    """The operations across the threads of a block of `threads` threads.

    threads: the `threads` of every kernel that calls them, a multiple of 32 up to 1024.
    """
    if threads % _WARP or not _WARP <= threads <= _MAX_THREADS:
        raise ValueError(f"a block of {threads} threads is not a multiple of 32 up to 1024")
    warps = threads // _WARP

    @device(threads=threads)
    def sum[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
        """The sum of `value` across the block, wrapping as `T` does."""
        partial = cuda.shared.array(warps, type(value))
        _staged(warp.sum(value), partial)
        total = partial[0]
        for index in range(1, warps):
            total = integers.wrapping_add(total, partial[index])
        cuda.syncthreads()
        return total

    @device(threads=threads)
    def min[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
        """The lowest `value` any thread holds."""
        partial = cuda.shared.array(warps, type(value))
        _staged(warp.min(value), partial)
        lowest = partial[0]
        for index in range(1, warps):
            lowest = builtins.min(lowest, partial[index])
        cuda.syncthreads()
        return lowest

    @device(threads=threads)
    def max[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
        """The highest `value` any thread holds."""
        partial = cuda.shared.array(warps, type(value))
        _staged(warp.max(value), partial)
        highest = partial[0]
        for index in range(1, warps):
            highest = builtins.max(highest, partial[index])
        cuda.syncthreads()
        return highest

    @device(threads=threads)
    def and_[T: (u32, u64)](value: T) -> T:
        """The bits `value` sets in every thread."""
        partial = cuda.shared.array(warps, type(value))
        _staged(warp.and_(value), partial)
        common = partial[0]
        for index in range(1, warps):
            common = type(value)(common & partial[index])
        cuda.syncthreads()
        return common

    @device(threads=threads)
    def or_[T: (u32, u64)](value: T) -> T:
        """The bits `value` sets in any thread."""
        partial = cuda.shared.array(warps, type(value))
        _staged(warp.or_(value), partial)
        any_set = partial[0]
        for index in range(1, warps):
            any_set = type(value)(any_set | partial[index])
        cuda.syncthreads()
        return any_set

    @device(threads=threads)
    def xor[T: (u32, u64)](value: T) -> T:
        """The bits `value` sets in an odd number of threads."""
        partial = cuda.shared.array(warps, type(value))
        _staged(warp.xor(value), partial)
        odd = partial[0]
        for index in range(1, warps):
            odd = type(value)(odd ^ partial[index])
        cuda.syncthreads()
        return odd

    @device(threads=threads)
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

    return Block(sum, min, max, and_, or_, xor, exclusive_sum)
