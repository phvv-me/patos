"""Integer arithmetic numba-cuda lacks, called through the module (`integers.ceildiv(n, 32)`)."""

from ..typed import device, dispatched, f32, f64, i32, i64, u32, u64


@device
def _ceildiv[T: (i32, u32, u64)](value: T, divisor: T) -> T:
    """Numba widens the 32-bit product to 64 bits, and an unsigned one never passes `value`."""
    quotient = value // divisor
    return type(value)(quotient + (quotient * divisor != value))


@device
def _ceildiv_i64(value: i64, divisor: i64) -> i64:
    """The remainder decides, since `quotient * divisor` passes `i64` min for a `value` there."""
    quotient, remainder = divmod(value, divisor)
    return quotient + (remainder != 0)


@dispatched(_ceildiv, _ceildiv_i64)
def ceildiv[T: (i32, u32, i64, u64)](value: T, divisor: T) -> T:
    """`value` divided by the positive `divisor`, rounded up.

    The floor quotient rises by one when it leaves a remainder, which needs no second division,
    and no sum overflows the way `(value + divisor - 1) // divisor` does near the top of `T`.
    """
    raise NotImplementedError


@device
def _wrapping_add[T: (i32, u32, u64, f32, f64)](a: T, b: T) -> T:
    """Numba adds 32-bit integers in 64 bits, which the conversion back wraps."""
    return type(a)(a + b)


@device
def _wrapping_add_i64(a: i64, b: i64) -> i64:
    """Through `u64`, whose sum Numba does not mark as never overflowing, as it does `i64`'s."""
    return u64(a) + b


@dispatched(_wrapping_add, _wrapping_add_i64)
def wrapping_add[T: (i32, u32, i64, u64, f32, f64)](a: T, b: T) -> T:
    """`a` plus `b` at their type, an integer wrapping around its range as an unsigned one does."""
    raise NotImplementedError
