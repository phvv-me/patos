"""Reductions and scans across the threads of one block, built on the warp's.

`over(threads)` gives the operations of blocks of that size, which is a compile-time constant so
that the loops over a block's warps unroll and the scratch holds one slot per warp. A kernel
launched with another size refuses to compile them. Every thread of the block calls them
together. A reduction takes the types its warp reduction does (`warp.sum` and the rest), so the
lowest non-negative value is `min(u32(value))` here too. Each stages one value per warp in shared
memory of its own, which a kernel that calls it holds once per type, and leaves the scratch free
for the next call.

`reduce`, `scan` and `exclusive` take any `Monoid` (`primitives.monoid`), and each named reduction
is the same engine at the monoid of its name, staging the warp reduction `warp.reduce` picks.
Every operation declares its own scratch and lends it to the engine, since numba-cuda refuses a
kernel that reaches one shared array through two device functions (NVVM fails to compile a kernel
calling `sum` and `reduce` of one block, both reaching the one scratch of `reduce`).
"""

from collections.abc import Callable
from functools import cache
from typing import NamedTuple, Protocol

from ..typed import Vector, cuda, device, f32, f64, i32, i64, lane, u32, u64, warp_in_block
from . import warp
from .monoid import Monoid, add, bitwise_and, bitwise_or, bitwise_xor, maximum, minimum

_WARP = 32
_MAX_THREADS = 1024


class Reduction(Protocol):
    """A block reduction of numbers, as a type checker reads it."""

    def __call__[T: (i32, u32, i64, u64, f32, f64)](self, value: T) -> T: ...


class Bitwise(Protocol):
    """A block reduction of bits, as a type checker reads it."""

    def __call__[T: (u32, u64)](self, value: T) -> T: ...


class Folding(Protocol):
    """A block reduction or inclusive scan by a monoid, as a type checker reads it."""

    def __call__[T: (i32, u32, i64, u64, f32, f64)](self, monoid: Monoid, value: T) -> T: ...


class Exclusive(Protocol):
    """A block exclusive scan by a monoid and the block's total, as a type checker reads it."""

    def __call__[T: (i32, u32, i64, u64, f32, f64)](
        self, monoid: Monoid, value: T
    ) -> tuple[T, T]: ...


class Block(NamedTuple):
    """The operations across the threads of a block of one size, every thread calling together."""

    sum: Reduction
    min: Reduction
    max: Reduction
    and_: Bitwise
    or_: Bitwise
    xor: Bitwise
    exclusive_sum: Callable[[i32], tuple[i32, i32]]
    reduce: Folding
    scan: Folding
    exclusive: Exclusive


@cache
def over(threads: int) -> Block:
    """The operations across the threads of a block of `threads` threads.

    threads: the `threads` of every kernel that calls them, a multiple of 32 up to 1024.
    """
    if threads % _WARP or not _WARP <= threads <= _MAX_THREADS:
        raise ValueError(f"a block of {threads} threads is not a multiple of 32 up to 1024")
    warps = threads // _WARP

    @device(threads=threads)
    def combined[M: Monoid, T: (i32, u32, i64, u64, f32, f64)](
        monoid: M, reduced: T, partial: Vector[T]
    ) -> T:
        """The `monoid` combination of every warp's `reduced`, held by every thread.

        partial: scratch of a slot per warp, left free.
        """
        if lane() == 0:
            partial[warp_in_block()] = reduced
        cuda.syncthreads()
        total = partial[0]
        for index in range(1, warps):
            total = monoid.combine(total, partial[index])
        cuda.syncthreads()
        return total

    @device(threads=threads)
    def preceding[M: Monoid, T: (i32, u32, i64, u64, f32, f64)](
        monoid: M, inclusive: T, totals: Vector[T]
    ) -> tuple[T, T]:
        """What the warps below this one combine to, and what the whole block does.

        inclusive: the warp's combination up to this lane, whose last lane stages the warp's.
        totals: scratch of a slot per warp, left free.
        """
        if lane() == 31:
            totals[warp_in_block()] = inclusive
        cuda.syncthreads()
        before = monoid.identity(inclusive)
        total = before
        for index in range(warps):
            if index < warp_in_block():
                before = monoid.combine(before, totals[index])
            total = monoid.combine(total, totals[index])
        cuda.syncthreads()
        return before, total

    @device(threads=threads)
    def reduce[M: Monoid, T: (i32, u32, i64, u64, f32, f64)](monoid: M, value: T) -> T:
        """The `monoid` combination of `value` across the block, whose operator commutes."""
        partial = cuda.shared.array(warps, type(value))
        return combined(monoid, warp.reduce(monoid, value), partial)

    @device(threads=threads)
    def scan[M: Monoid, T: (i32, u32, i64, u64, f32, f64)](monoid: M, value: T) -> T:
        """The `monoid` combination of `value` across the threads up to this one, this included."""
        totals = cuda.shared.array(warps, type(value))
        inclusive = warp.scan(monoid, value)
        before, _ = preceding(monoid, inclusive, totals)
        return monoid.combine(before, inclusive)

    @device(threads=threads)
    def exclusive[M: Monoid, T: (i32, u32, i64, u64, f32, f64)](
        monoid: M, value: T
    ) -> tuple[T, T]:
        """The `monoid` combination of `value` across the threads below this one, and the total."""
        totals = cuda.shared.array(warps, type(value))
        within = warp.exclusive(monoid, value)
        before, total = preceding(monoid, monoid.combine(within, value), totals)
        return monoid.combine(before, within), total

    @device(threads=threads)
    def sum[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
        """The sum of `value` across the block, wrapping as `T` does."""
        partial = cuda.shared.array(warps, type(value))
        return combined(add, warp.sum(value), partial)

    @device(threads=threads)
    def min[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
        """The lowest `value` any thread holds."""
        partial = cuda.shared.array(warps, type(value))
        return combined(minimum, warp.min(value), partial)

    @device(threads=threads)
    def max[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
        """The highest `value` any thread holds."""
        partial = cuda.shared.array(warps, type(value))
        return combined(maximum, warp.max(value), partial)

    @device(threads=threads)
    def and_[T: (u32, u64)](value: T) -> T:
        """The bits `value` sets in every thread."""
        partial = cuda.shared.array(warps, type(value))
        return combined(bitwise_and, warp.and_(value), partial)

    @device(threads=threads)
    def or_[T: (u32, u64)](value: T) -> T:
        """The bits `value` sets in any thread."""
        partial = cuda.shared.array(warps, type(value))
        return combined(bitwise_or, warp.or_(value), partial)

    @device(threads=threads)
    def xor[T: (u32, u64)](value: T) -> T:
        """The bits `value` sets in an odd number of threads."""
        partial = cuda.shared.array(warps, type(value))
        return combined(bitwise_xor, warp.xor(value), partial)

    @device(threads=threads)
    def exclusive_sum(value: i32) -> tuple[i32, i32]:
        """`value`'s exclusive prefix sum across the block and the block's total."""
        totals = cuda.shared.array(warps, i32)
        inclusive = warp.inclusive_sum(value)
        before, total = preceding(add, inclusive, totals)
        return before + inclusive - value, total

    return Block(sum, min, max, and_, or_, xor, exclusive_sum, reduce, scan, exclusive)
