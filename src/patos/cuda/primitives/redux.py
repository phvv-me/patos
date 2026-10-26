"""The warp reductions that run on `redux.sync`, which `warp.reduce` takes where it is faster.

`redux.sync` reduces a 32-bit integer of the whole warp in one instruction (sm_80 and newer),
where a butterfly shuffles five times. `strategy` names, for a monoid and the type of its operand,
the device function that reduces by it, or None where the butterfly stays. Each strategy is exact:

- a 32-bit integer is one `redux.sync` of the operator's own kind;
- the bitwise operators never cross a bit, so a 64-bit integer is two, one on each half;
- the minimum and maximum of a 64-bit integer reduce the high halves, and then the low halves of
  the lanes holding that high half, the others passing the identity;
- the sum of a 64-bit integer is exact modulo 2^64: the low word's two 16-bit halves each sum
  inside 32 bits, and the high word's sum wraps as the word does;
- a float's minimum and maximum reduce an integer key that orders as the float does (a negative
  float's bits flipped, a positive one's sign bit set), so `-0.0` is below `+0.0`, and a NaN is
  a lane with nothing to give, passing the identity: it is chosen only where every lane holds one.

The sum of floats stays a butterfly: its fixed tree keeps results the same on every run.
"""

from collections.abc import Callable, Iterator, Sequence
from functools import cache
from typing import overload

import numpy as np
from numba import typeof, types
from numba.cuda import libdevice

from ..typed import device, dispatched, f32, f64, i32, i64, ptx, selected, u32, u64
from . import bits, monoid
from .monoid import Monoid


@ptx("redux.sync.add.s32 $result, $value, 0xffffffff;")
def _add_i32(value: i32) -> i32:
    raise NotImplementedError


@ptx("redux.sync.add.u32 $result, $value, 0xffffffff;")
def _add_u32(value: u32) -> u32:
    raise NotImplementedError


@dispatched(_add_i32, _add_u32)
def _add[T: (i32, u32)](value: T) -> T:
    raise NotImplementedError


@ptx("redux.sync.min.s32 $result, $value, 0xffffffff;")
def _min_i32(value: i32) -> i32:
    raise NotImplementedError


@ptx("redux.sync.min.u32 $result, $value, 0xffffffff;")
def _min_u32(value: u32) -> u32:
    raise NotImplementedError


@dispatched(_min_i32, _min_u32)
def _lowest[T: (i32, u32)](value: T) -> T:
    raise NotImplementedError


@ptx("redux.sync.max.s32 $result, $value, 0xffffffff;")
def _max_i32(value: i32) -> i32:
    raise NotImplementedError


@ptx("redux.sync.max.u32 $result, $value, 0xffffffff;")
def _max_u32(value: u32) -> u32:
    raise NotImplementedError


@dispatched(_max_i32, _max_u32)
def _highest[T: (i32, u32)](value: T) -> T:
    raise NotImplementedError


@ptx("redux.sync.and.b32 $result, $value, 0xffffffff;")
def _and(value: u32) -> u32:
    raise NotImplementedError


@ptx("redux.sync.or.b32 $result, $value, 0xffffffff;")
def _or(value: u32) -> u32:
    raise NotImplementedError


@ptx("redux.sync.xor.b32 $result, $value, 0xffffffff;")
def _xor(value: u32) -> u32:
    raise NotImplementedError


@device
def _upper_i64(value: i64) -> i32:
    return value >> 32


@device
def _upper_u64(value: u64) -> u32:
    return value >> 32


@overload
def _upper(value: i64) -> i32: ...
@overload
def _upper(value: u64) -> u32: ...
@dispatched(_upper_i64, _upper_u64)
def _upper(value: i64 | u64) -> i32 | u32:
    """The high word of `value`, signed where `value` is."""
    raise NotImplementedError


def _smaller(operands: Sequence[types.Type]) -> Callable:
    """The strategy for the monoid and integer a strategy of a wider type reduces its parts by."""
    operator, value = operands
    return _STRATEGIES[named()[operator], types.unliteral(value)]


@selected(_smaller)
def _reduced[M: Monoid, T: (i32, u32, i64, u64)](operator: M, value: T) -> T:
    """What `operator` reduces `value` to, by the strategy of its own type."""
    raise NotImplementedError


@device
def _sum_word[M: Monoid, T: (i32, u32)](operator: M, value: T) -> T:
    return type(value)(_add(value))


@device
def _sum_wide[M: Monoid, T: (i64, u64)](operator: M, value: T) -> T:
    low: u32 = value
    masked: u32 = low & 0xFFFF
    shifted: u32 = low >> 16
    top: u32 = value >> 32
    small: u32 = _add(masked)
    large: u32 = _add(shifted)
    high: u32 = _add(top)
    return type(value)(small + (u64(large) << 16) + (u64(high) << 32))


@device
def _lowest_word[M: Monoid, T: (i32, u32)](operator: M, value: T) -> T:
    return _lowest(value)


@device
def _highest_word[M: Monoid, T: (i32, u32)](operator: M, value: T) -> T:
    return _highest(value)


@device
def _extreme_wide[M: Monoid, T: (i64, u64)](operator: M, value: T) -> T:
    top = _upper(value)
    best = _reduced(operator, top)
    low: u32 = value
    if top != best:
        low = operator.identity(low)
    high: u32 = best
    return type(value)(bits.join(low=_reduced(operator, low), high=high))


@device
def _and_word[M: Monoid, T: (i32, u32)](operator: M, value: T) -> T:
    return type(value)(_and(value))


@device
def _or_word[M: Monoid, T: (i32, u32)](operator: M, value: T) -> T:
    return type(value)(_or(value))


@device
def _xor_word[M: Monoid, T: (i32, u32)](operator: M, value: T) -> T:
    return type(value)(_xor(value))


@device
def _halves[M: Monoid, T: (i64, u64)](operator: M, value: T) -> T:
    low: u32 = value
    high: u32 = value >> 32
    return type(value)(bits.join(low=_reduced(operator, low), high=_reduced(operator, high)))


@device
def _ordered[T: (i32, i64)](raw: T) -> T:
    """The bits of a float flipped so that, read unsigned, they order as the float does."""
    sign = np.iinfo(type(raw)).min
    return type(raw)(raw ^ ((raw >> (np.iinfo(type(raw)).bits - 1)) | sign))


@device
def _unordered[T: (i32, i64)](raw: T) -> T:
    """The bits of the float that `_ordered` made `raw` of."""
    sign = np.iinfo(type(raw)).min
    return type(raw)(raw ^ (~(raw >> (np.iinfo(type(raw)).bits - 1)) | sign))


@device
def _extreme_f32[M: Monoid](operator: M, value: f32) -> f32:
    key = u32(_ordered(libdevice.float_as_int(value)))
    if value != value:
        key = operator.identity(key)
    return libdevice.int_as_float(_unordered(i32(_reduced(operator, key))))


@device
def _extreme_f64[M: Monoid](operator: M, value: f64) -> f64:
    key = u64(_ordered(libdevice.double_as_longlong(value)))
    if value != value:
        key = operator.identity(key)
    return libdevice.longlong_as_double(_unordered(i64(_reduced(operator, key))))


# Each built-in monoid's strategy for a 32-bit integer, a 64-bit integer, an f32 and an f64.
_BY_MONOID: dict[str, tuple[Callable | None, ...]] = {
    "add": (_sum_word, _sum_wide, None, None),
    "minimum": (_lowest_word, _extreme_wide, _extreme_f32, _extreme_f64),
    "maximum": (_highest_word, _extreme_wide, _extreme_f32, _extreme_f64),
    "bitwise_and": (_and_word, _halves, None, None),
    "bitwise_or": (_or_word, _halves, None, None),
    "bitwise_xor": (_xor_word, _halves, None, None),
}
_KINDS = (
    (types.int32, 0), (types.uint32, 0), (types.int64, 1), (types.uint64, 1),
    (types.float32, 2), (types.float64, 3),
)  # fmt: skip
_STRATEGIES: dict[tuple[str, types.Type], Callable] = {
    (name, kind): found
    for name, strategies in _BY_MONOID.items()
    for kind, place in _KINDS
    if (found := strategies[place])
}


@cache
def named() -> dict[types.Type, str]:
    """The name of each built-in monoid by the Numba type of its value."""
    return {typeof(getattr(monoid, name)): name for name in _BY_MONOID}


def served() -> Iterator[tuple[str, types.Type]]:
    """The monoid name and type of each reduction a strategy serves."""
    return iter(_STRATEGIES)


def strategy(operands: Sequence[types.Type]) -> Callable | None:
    """The device function reducing by the monoid typed `operands[0]` a value typed `operands[1]`.

    None where no strategy is faster than the butterfly.
    """
    operator, value = operands
    name = named().get(operator)
    return None if name is None else _STRATEGIES.get((name, types.unliteral(value)))
