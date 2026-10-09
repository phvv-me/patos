"""Integer arithmetic numba-cuda lacks, called through the module (`integers.ceildiv(n, 32)`)."""

from ..typed import device, dispatched, i32, i64, u32, u64


@device
def _ceildiv[T: (i32, u32, i64, u64)](value: T, divisor: T) -> T:
    quotient = value // divisor
    return type(value)(quotient + (quotient * divisor != value))


@dispatched(_ceildiv)
def ceildiv[T: (i32, u32, i64, u64)](value: T, divisor: T) -> T:
    """`value` divided by the positive `divisor`, rounded up.

    The floor quotient rises by one when it leaves a remainder, which needs no second division,
    and no sum overflows the way `(value + divisor - 1) // divisor` does near the top of `T`.
    """
    raise NotImplementedError
