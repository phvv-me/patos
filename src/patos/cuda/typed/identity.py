"""Who a device function runs as, and how many run with it: its thread, warp, lane and block.

A device function reads its identity here instead of taking a `lane` or a `thread` parameter.
Each helper inlines to the expression it names, so using one costs what writing the expression
does. A warp is 32 threads, stated as a literal so that dividing by it types as the thread index.
"""

from numba import cuda

from ..scalars import i32, i64
from .decorators import device


@device
def thread_index() -> i64:
    """The thread's index in the grid."""
    return cuda.grid(1)


@device
def warp_index() -> i64:
    """The index in the grid of the warp the thread is in."""
    return cuda.grid(1) // 32


@device
def block_index() -> i32:
    """The index in the grid of the block the thread is in."""
    return cuda.blockIdx.x


@device
def lane() -> i32:
    """The thread's lane in its warp."""
    return cuda.laneid


@device
def thread_in_block() -> i32:
    """The thread's index in its block."""
    return cuda.threadIdx.x


@device
def warp_in_block() -> i32:
    """The index in its block of the warp the thread is in."""
    return cuda.threadIdx.x // 32


@device
def thread_count() -> i64:
    """How many threads the grid holds."""
    return cuda.gridsize(1)


@device
def warp_count() -> i64:
    """How many warps the grid holds."""
    return cuda.gridsize(1) // 32


@device
def block_count() -> i32:
    """How many blocks the grid holds."""
    return cuda.gridDim.x
