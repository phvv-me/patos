"""The warp-wide operations numba-cuda lacks, all over the full 32-lane mask.

Call them through the module (`warp.sum(x)`), so `sum` shadows nothing at the call site. A sum and
a lowest value are one `redux.sync` instruction on sm_80 and newer, which numba-cuda does not
expose and libNVVM lowers only for some targets, so they are PTX. The scan is CUB's: each step's
shuffle also reports whether the source lane exists, so the add needs no lane test.
"""

from ..typed import cuda, device, i32, ptx

# The lane mask naming all 32 lanes of a warp.
_FULL_WARP = 0xFFFFFFFF
# One step per offset: shuffle the running total up, and add it where a lane that far down exists.
_SCAN_STEPS = "".join(
    f"shfl.sync.up.b32 received|exists, total, {offset}, 0, {_FULL_WARP:#x};\n"
    "@exists add.s32 total, total, received;\n"
    for offset in (1, 2, 4, 8, 16)
)


@ptx(f"redux.sync.add.s32 $result, $value, {_FULL_WARP:#x};")
def sum(value: i32) -> i32:
    """Sum `value` across the warp, every lane receiving the total."""
    ...


@ptx(f"redux.sync.min.u32 $result, $value, {_FULL_WARP:#x};")
def min_nonnegative(value: i32) -> i32:
    """The lowest non-negative `value` any lane holds, or -1 when every lane holds -1.

    -1 reads as the largest u32, so the unsigned minimum is the lowest non-negative value.
    """
    ...


@ptx(
    "{\n.reg .s32 total, received;\n.reg .pred exists;\nmov.s32 total, $value;\n"
    + _SCAN_STEPS
    + "mov.s32 $result, total;\n}"
)
def inclusive_sum(value: i32) -> i32:
    """The inclusive prefix sum of `value` across the warp."""
    ...


@device
def ballot_prefix(flag: bool) -> tuple[i32, i32]:
    """Return how many lower lanes set `flag` and how many lanes set it in all.

    The pair is what a compaction needs: a lane that set the flag writes at the first count
    behind whatever was written before, and the warp advances by the second.
    """
    found = cuda.ballot_sync(_FULL_WARP, flag) & _FULL_WARP
    return cuda.popc(found & cuda.lanemask_lt()), cuda.popc(found)
