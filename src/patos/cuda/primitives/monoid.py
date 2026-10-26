"""Associative operators with identities, which the warp and block engines reduce and scan by.

A `Monoid` is a named value of two device functions that device code passes on as it is:
`warp.reduce(maximum, value)`, `block.over(256).scan(add, value)`, named as numpy names the ufuncs
whose `reduce` and `accumulate` they are. Every engine takes `i32`, `u32`, `i64`, `u64`, `f32` and
`f64`. A reduction also needs `combine` to commute, a scan only to associate, and the identity
serves the exclusive scans. Packed lanes travel as the `u32` word they fill, so `bitwise_and`,
`bitwise_or` and `bitwise_xor` reduce them lane by lane, never crossing a lane, and
`bits.bytes_min` and `bits.bytes_max` do so for the bytes of a word. Any other lane-wise operator
is a `Monoid` over words, its `combine` converting them (`u8x4(word)`) to the operation in `bits`.

The built-in monoids below are known to the engines: `warp.reduce` reduces each of them by the
fastest exact way for the type (`redux`),
and `wide` runs them over a whole device array on cuda.compute's own operators. Any other monoid
takes the shuffles of the warp and block engines, and is refused by `wide`.
"""

import builtins
import math
from typing import NamedTuple, Protocol

import numpy as np

from ..typed import device, dispatched, f32, f64, i32, i64, u32, u64
from . import integers


class Combine(Protocol):
    """An operator over two values of one type, as a type checker reads it."""

    def __call__[T: (i32, u32, i64, u64, f32, f64)](self, a: T, b: T) -> T: ...


class Identity(Protocol):
    """The value of `like`'s type an operator leaves any other as it is, as a checker reads it."""

    def __call__[T: (i32, u32, i64, u64, f32, f64)](self, like: T) -> T: ...


class Monoid(NamedTuple):
    """An associative operator and its identity, device functions over any type they take.

    combine: the operator, `combine(a, b)` for the value `a` of a lower lane and `b` of a higher.
    identity: `identity(like)` is the value of `like`'s type that `combine` leaves unchanged.
    """

    combine: Combine
    identity: Identity


@device
def zero[T: (i32, u32, i64, u64, f32, f64)](like: T) -> T:
    """Zero of `like`'s type, which a sum, `OR` and `XOR` leave unchanged."""
    return type(like)(0)


@device
def ones[T: (i32, u32, i64, u64)](like: T) -> T:
    """Every bit of `like`'s type set, which `AND` leaves unchanged."""
    return type(like)(-1)


@device
def _highest_integer[T: (i32, u32, i64, u64)](like: T) -> T:
    return type(like)(np.iinfo(type(like)).max)


@device
def _highest_float[T: (f32, f64)](like: T) -> T:
    return type(like)(math.inf)


@dispatched(_highest_integer, _highest_float)
def highest[T: (i32, u32, i64, u64, f32, f64)](like: T) -> T:
    """The highest value of `like`'s type, which a minimum leaves unchanged."""
    raise NotImplementedError


@device
def _lowest_integer[T: (i32, u32, i64, u64)](like: T) -> T:
    return type(like)(np.iinfo(type(like)).min)


@device
def _lowest_float[T: (f32, f64)](like: T) -> T:
    return type(like)(-math.inf)


@dispatched(_lowest_integer, _lowest_float)
def lowest[T: (i32, u32, i64, u64, f32, f64)](like: T) -> T:
    """The lowest value of `like`'s type, which a maximum leaves unchanged."""
    raise NotImplementedError


@device
def _sum[T: (i32, u32, i64, u64, f32, f64)](a: T, b: T) -> T:
    return integers.wrapping_add(a, b)


@device
def _minimum[T: (i32, u32, i64, u64, f32, f64)](a: T, b: T) -> T:
    return builtins.min(a, b)


@device
def _maximum[T: (i32, u32, i64, u64, f32, f64)](a: T, b: T) -> T:
    return builtins.max(a, b)


@device
def _and[T: (i32, u32, i64, u64)](a: T, b: T) -> T:
    return type(a)(a & b)


@device
def _or[T: (i32, u32, i64, u64)](a: T, b: T) -> T:
    return type(a)(a | b)


@device
def _xor[T: (i32, u32, i64, u64)](a: T, b: T) -> T:
    return type(a)(a ^ b)


add = Monoid(_sum, zero)
minimum = Monoid(_minimum, highest)
maximum = Monoid(_maximum, lowest)
bitwise_and = Monoid(_and, ones)
bitwise_or = Monoid(_or, zero)
bitwise_xor = Monoid(_xor, zero)
