"""The warp-wide operations numba-cuda lacks, all over the full 32-lane mask.

Call them through the module (`warp.sum(x)`), so `sum` shadows nothing at the call site. All 32
lanes of the warp call one together, a lane with nothing to give passing the neutral value. A
reduction is one `redux.sync` instruction on sm_80 and newer, which numba-cuda does not expose
and libNVVM lowers only for some targets, so each is PTX. The scan is CUB's: each step's shuffle
also reports whether the source lane exists, so the add needs no lane test.
"""

from collections.abc import Callable

from ..typed import Vector, cuda, device, i32, lane, ptx, u32

# The lane mask naming all 32 lanes of a warp.
_FULL_WARP = 0xFFFFFFFF
# One step per offset: shuffle the running total up, and add it where a lane that far down exists.
_SCAN_STEPS = "".join(
    f"shfl.sync.up.b32 received|exists, total, {offset}, 0, {_FULL_WARP:#x};\n"
    "@exists add.s32 total, total, received;\n"
    for offset in (1, 2, 4, 8, 16)
)


def _reduction[T](name: str, kind: type[T], *, operation: str, doc: str) -> Callable[[T], T]:
    """The warp reduction `redux.sync.<operation>`, taking and giving `kind`.

    name: the name the function answers to.
    operation: the PTX operation and type, which `kind` follows in signedness.
    doc: what it computes, every lane receiving it.
    """

    def stub(value: T) -> T: ...

    stub.__annotations__ = {"value": kind, "return": kind}
    stub.__name__ = stub.__qualname__ = name
    stub.__doc__ = doc
    return ptx(f"redux.sync.{operation} $result, $value, {_FULL_WARP:#x};")(stub)


sum = _reduction("sum", i32, operation="add.s32", doc="The sum of `value` across the warp.")
min = _reduction("min", i32, operation="min.s32", doc="The lowest `value` any lane holds.")
max = _reduction("max", i32, operation="max.s32", doc="The highest `value` any lane holds.")
min_unsigned = _reduction(
    "min_unsigned",
    u32,
    operation="min.u32",
    doc="The lowest `value` any lane holds, read unsigned.",
)
max_unsigned = _reduction(
    "max_unsigned",
    u32,
    operation="max.u32",
    doc="The highest `value` any lane holds, read unsigned.",
)
all_bits = _reduction(
    "all_bits", u32, operation="and.b32", doc="The bits `value` sets in every lane."
)
any_bits = _reduction(
    "any_bits", u32, operation="or.b32", doc="The bits `value` sets in any lane."
)
odd_bits = _reduction(
    "odd_bits", u32, operation="xor.b32", doc="The bits `value` sets in an odd number of lanes."
)
min_nonnegative = _reduction(
    "min_nonnegative", i32, operation="min.u32",
    doc="""The lowest non-negative `value` any lane holds, or -1 when every lane holds -1.

    -1 reads as the largest u32, so the unsigned minimum is the lowest non-negative value.
    """,
)  # fmt: skip


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


@device
def first(mask: u32) -> i32:
    """The lowest lane set in `mask`, or -1 when none is."""
    return cuda.ffs(mask) - 1


@device
def reserve(counter: Vector[i32], flag: bool) -> i32:
    """The slot each lane that set `flag` claims at the end of `counter[0]`, -1 for the others.

    A lane with nothing to append passes False. The flagged lanes of the warp take consecutive
    slots in lane order with one atomic add, made by the lowest of them and broadcast to the rest.
    """
    found = cuda.ballot_sync(_FULL_WARP, flag) & _FULL_WARP
    slot: i32 = -1
    if found != 0:
        leader = first(found)
        base: i32 = 0
        if lane() == leader:
            base = cuda.atomic.add(counter, 0, cuda.popc(found))
        base = cuda.shfl_sync(_FULL_WARP, base, leader)
        if flag:
            slot = base + cuda.popc(found & cuda.lanemask_lt())
    return slot
