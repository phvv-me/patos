"""Offsets handed out in order by a decoupled look-back, inside the kernel that counts them.

A kernel that counts what each thread writes, and then writes it, needs no scan launch between the
two. Its blocks take tiles of the work in order (`claim`), each block scans its tile and publishes
the tile's total, and the tile looks back over the totals of the tiles before it for what they hold
(`before`). Every tile publishes its total before it looks back, so the wait ends, and a block only
waits on tiles that blocks already running hold.

    ordered = offsets.over(256)

    @kernel(per=Per.BLOCK, threads=256, strided=True)
    def place(counts: Vector[i32], starts: Vector[i32], chain: offsets.Chain) -> None:
        while True:
            tile = ordered.claim(chain)
            if tile * 256 >= starts.size - 1:
                return
            index = tile * 256 + thread_in_block()
            count: i32 = 0
            if index < starts.size - 1:
                count = counts[index]
            ordered.counted(chain, tile, index, count, starts)

    chain = offsets.chain(workspace, blocks, stream=stream)
    place[blocks](counts, starts, chain, stream=stream)

`counted` enters each count in `starts` as the sum of the counts before it, and the total last. The
general form is `place`, which hands each thread the combination of the counts before it by any
`Monoid` over a 32-bit integer (a tile's status is one 64-bit word), and the combination up to the
end of its tile, which is the whole of it in the last one.

`scan` applies the same machinery to a whole device array: one block for an array a tile holds,
and one pass of many, ordered by the look-back, for the rest.
"""

from functools import cache
from typing import TYPE_CHECKING, NamedTuple, Protocol, Self

import numpy as np

from ..runtime.memory import Workspace
from ..scalars import Shaped
from ..typed import (
    Constant,
    Per,
    Struct,
    Vector,
    cuda,
    device,
    i32,
    i64,
    kernel,
    lane,
    thread_in_block,
    u32,
    warp_in_block,
)
from . import block, warp
from .monoid import Monoid, add

if TYPE_CHECKING:
    from ..runtime.streams import StreamLike

# The look-back state of one ordered pass: the status of each tile, then the ticket that hands the
# blocks their tiles in order. A status holds a value in its low word and, above it, whether that
# is the tile's own total or its inclusive prefix; zero is a tile that has published nothing yet.
Chain = Vector[i64]

_TOTAL = 1 << 32
_PREFIX = 2 << 32
_VALUE = (1 << 32) - 1
_WARP = 32
_THREADS = 256
# Elements a lane takes, in runs of 32, of an array up to `_VAST` long and of a longer one.
_ROWS = (4, 16)
_VAST = 1 << 18


def chain(
    workspace: Workspace, tiles: int, *, stream: StreamLike, role: str = "patos.offsets.chain"
) -> Chain:
    """A look-back state for `tiles` tiles, taken from `workspace` under `role`.

    It is zeroed by work queued on `stream`.
    """
    return workspace.zeros(role, tiles + 1, np.int64, stream=stream)


class Viewable(Shaped, Protocol):
    """An array that reads its bytes as another element type, as numpy's and CuPy's do."""

    def view(self, dtype: type[np.generic]) -> Self: ...


class Claiming(Protocol):
    """A block's next tile, as a type checker reads it."""

    def __call__(self, chain: Chain) -> i32: ...


class Looking(Protocol):
    """What the tiles before this one hold, as a type checker reads it."""

    def __call__[T: (i32, u32)](self, monoid: Monoid, chain: Chain, tile: i32, total: T) -> T: ...


class Placing(Protocol):
    """A thread's offset and its tile's end, as a type checker reads it."""

    def __call__[T: (i32, u32)](
        self, monoid: Monoid, chain: Chain, tile: i32, count: T
    ) -> tuple[T, T]: ...


class Counting(Protocol):
    """A thread's count entered in offsets, as a type checker reads it."""

    def __call__(
        self, chain: Chain, tile: i32, index: i32, count: i32, starts: Vector[i32]
    ) -> None: ...


class Ordered(NamedTuple):
    """The ordered operations of blocks of one size, every thread of the block calling together."""

    claim: Claiming
    before: Looking
    place: Placing
    counted: Counting


@device
def _status[T: (i32, u32)](flag: i64, value: T) -> i64:
    return flag | (i64(value) & _VALUE)


@device
def _published(chain: Chain, tile: i32) -> i64:
    """Tile `tile`'s status, once it has published one."""
    status: i64 = cuda.atomic.add(chain, tile, 0)
    while status == 0:
        status = cuda.atomic.add(chain, tile, 0)
    return status


@device
def _window[M: Monoid, T: (i32, u32)](
    monoid: M, chain: Chain, window: i32, like: T
) -> tuple[T, u32]:
    """What the 32 tiles down from `window` hold up to the nearest prefix, and which hold one."""
    predecessor = window - lane()
    status: i64 = _status(_PREFIX, monoid.identity(like))
    if predecessor >= 0:
        status = _published(chain, predecessor)
    prefixed = warp.ballot(status >= _PREFIX)
    held = type(like)(status & _VALUE)
    if prefixed != 0 and lane() > warp.first(prefixed):
        held = monoid.identity(like)
    return warp.reduce(monoid, held), prefixed


@device
def _looked_back[M: Monoid, T: (i32, u32)](monoid: M, chain: Chain, tile: i32, total: T) -> T:
    """Publish `total` as tile `tile`'s own, returning what the tiles before it hold; a warp's.

    A warp reads 32 predecessors at once, nearest first, back to the nearest holding its
    inclusive prefix, which the tile then publishes in place of its total.
    """
    if lane() == 0:
        cuda.atomic.exch(chain, tile, _status(_TOTAL, total))
    ahead = monoid.identity(total)
    window: i32 = tile - 1
    prefixed: u32 = 0
    while prefixed == 0:
        held, prefixed = _window(monoid, chain, window, total)
        ahead = monoid.combine(held, ahead)
        window -= _WARP
    if lane() == 0:
        cuda.atomic.exch(chain, tile, _status(_PREFIX, monoid.combine(ahead, total)))
    return ahead


@cache
def over(threads: int) -> Ordered:
    """The ordered operations of blocks of `threads` threads, made once per block size.

    threads: the `threads` of every kernel that calls them, a multiple of 32 up to 1024.
    """
    across = block.over(threads)

    @device(threads=threads)
    def claim(chain: Chain) -> i32:
        """The tile this block takes next, as every thread of the block reads it.

        Tiles come in order, so a tile only waits on tiles that blocks already hold. A barrier
        must lie between two claims, which `place` and `before` provide.
        """
        taken = cuda.shared.array(1, i32)
        if thread_in_block() == 0:
            taken[0] = cuda.atomic.add(chain, chain.size - 1, 1)
        cuda.syncthreads()
        return taken[0]

    @device(threads=threads)
    def before[M: Monoid, T: (i32, u32)](monoid: M, chain: Chain, tile: i32, total: T) -> T:
        """Publish tile `tile`'s `total`, returning to every thread what the tiles before it hold.

        total: what the whole block's counts combine to, as every thread holds it.
        """
        ahead = cuda.shared.array(1, type(total))
        held = monoid.identity(total)
        if thread_in_block() < _WARP:
            held = _looked_back(monoid, chain, tile, total)
        if thread_in_block() == 0:
            ahead[0] = held
        cuda.syncthreads()
        return ahead[0]

    @device(threads=threads)
    def place[M: Monoid, T: (i32, u32)](
        monoid: M, chain: Chain, tile: i32, count: T
    ) -> tuple[T, T]:
        """The combination of every count before this thread's, and up to the end of its tile.

        count: what this thread counts, the identity where it has nothing.
        """
        within, total = across.exclusive(monoid, count)
        ahead = before(monoid, chain, tile, total)
        return monoid.combine(ahead, within), monoid.combine(ahead, total)

    @device(threads=threads)
    def counted(chain: Chain, tile: i32, index: i32, count: i32, starts: Vector[i32]) -> None:
        """Enter `count` in `starts`, where each index gets the sum of the counts before it.

        index: the entry this thread counts, or any index past the last for a thread with none.
        starts: one entry more than there are indices, the extra one the total, which the thread
            of the last index writes. A thread past the last index counts zero.
        """
        start, _ = place(add, chain, tile, count)
        last = starts.size - 1
        if index < last:
            starts[index] = start
        if index == last - 1:
            starts[last] = start + count

    return Ordered(claim, before, place, counted)


_ACROSS = block.over(_THREADS)
_ORDERED = over(_THREADS)


class Scan(Struct):
    """A scan of an array by the sum, each warp taking `rows` runs of 32 consecutive elements, so
    that every load and store is coalesced, scanning each run by shuffles and carrying the sum of
    the runs before it to the next.

    rows: the runs a warp takes.
    inclusive: whether a scan includes the element it is written at.
    chain: the look-back state of a scan of many tiles, which one tile leaves alone.
    """

    rows: Constant[int]
    inclusive: Constant[bool]
    chain: Chain

    @device
    def first(self, tile: i32) -> i32:
        """Where this lane's first run begins in tile `tile`."""
        return (tile * _THREADS + warp_in_block() * _WARP) * self.rows + lane()

    @device
    def held(self, source: Vector[i32], first: i32) -> i32:
        """What the warp's runs sum to, as every lane holds it."""
        total: i32 = 0
        for row in range(self.rows):
            if first + row * _WARP < source.size:
                total += source[first + row * _WARP]
        return warp.sum(total)

    @device
    def written(self, source: Vector[i32], target: Vector[i32], first: i32, start: i32) -> None:
        """Write the warp's runs scanned from `start`."""
        carry = start
        for row in range(self.rows):
            slot = first + row * _WARP
            value: i32 = 0
            if slot < source.size:
                value = source[slot]
            running = warp.scan(add, value)
            if slot < source.size:
                target[slot] = carry + (running if self.inclusive else running - value)
            carry += warp.broadcast(running, _WARP - 1)

    @kernel(per=Per.BLOCK, threads=_THREADS)
    def single(self, source: Vector[i32], target: Vector[i32]) -> None:
        """Scan an array of one tile, a block of its own and nothing else."""
        first = self.first(0)
        total = self.held(source, first)
        start, _ = _ACROSS.exclusive(add, total if lane() == 0 else 0)
        self.written(source, target, first, warp.broadcast(start, 0))

    @kernel(per=Per.BLOCK, threads=_THREADS, strided=True)
    def chained(self, source: Vector[i32], target: Vector[i32]) -> None:
        """Scan an array of many tiles, `target` being `source` itself for a scan in place."""
        while True:
            tile = _ORDERED.claim(self.chain)
            if tile * _THREADS * self.rows >= source.size:
                return
            first = self.first(tile)
            total = self.held(source, first)
            start, _ = _ORDERED.place(add, self.chain, tile, total if lane() == 0 else 0)
            self.written(source, target, first, warp.broadcast(start, 0))


def scan[A: Viewable](
    values: A, *, out: A, workspace: Workspace, stream: StreamLike, inclusive: bool = True
) -> A:
    """Write into `out` the sums of `values` up to each, itself included or not, wrapping.

    A 32-bit integer array; `out` is `values` itself for a scan in place. An array a tile holds is
    scanned by one block, with no scratch; a larger one by a single pass of many, either queued on
    `stream`.
    """
    size = values.shape[0]
    rows = _ROWS[size > _VAST]
    ints, results = values.view(np.int32), out.view(np.int32)
    if size <= _THREADS * rows:
        single = Scan(rows, inclusive, workspace.take("patos.offsets.idle", 1, np.int64))
        single.single[1](ints, results, stream=stream)
    else:
        tiles = -(-size // (_THREADS * rows))
        chained = Scan(rows, inclusive, chain(workspace, tiles, stream=stream))
        chained.chained[tiles](ints, results, stream=stream)
    return out
