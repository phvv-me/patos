"""One device function name over typed implementations: what is refused at import when they drift
from the stub, and how a call with an integer literal is picked."""

from collections.abc import Callable
from typing import overload

import cupy as cp
import numpy as np
import pytest
from numba.core.errors import TypingError

from patos.cuda.primitives import bits, warp
from patos.cuda.typed import (
    Vector,
    device,
    dispatched,
    f32,
    f64,
    i32,
    i64,
    kernel,
    thread_index,
    u8,
    u32,
    u64,
)

gpu = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")


@device
def _kept[T: (i32, u32)](value: T) -> T:
    return value


@device
def _alike[T: (i32, u32)](first: T, flag: u8, second: T) -> T:
    return first


@device
def _widened(value: i32) -> i64:
    return value


def _constraint_short_of_the_signature() -> None:
    def total[T: (i32, u32, f64)](value: T) -> T:
        raise NotImplementedError

    dispatched(_kept)(total)


def _parameters_that_share_a_type_parameter() -> None:
    @overload
    def pair(first: i32, flag: u8, second: i32) -> i32: ...
    @overload
    def pair(first: i32, flag: u8, second: u32) -> i32: ...
    def pair(first: i32, flag: u8, second: i32 | u32) -> i32:
        raise NotImplementedError

    dispatched(_alike)(pair)


def _return_other_than_the_stub_declares() -> None:
    def widen(value: i32) -> i32:
        raise NotImplementedError

    dispatched(_widened)(widen)


def _overload_of_another_arity() -> None:
    @overload
    def narrow(value: i32) -> i32: ...
    @overload
    def narrow(value: i32, other: u8) -> i32: ...
    def narrow(value: i32, other: u8 = 0) -> i32:
        raise NotImplementedError

    dispatched(_kept)(narrow)


def _signature_declared_twice() -> None:
    @overload
    def twice(value: i32) -> i32: ...
    @overload
    def twice(value: i32) -> i32: ...
    def twice(value: i32) -> i32:
        raise NotImplementedError

    dispatched(_kept)(twice)


def _four_independent_type_parameters() -> None:
    """Six constraints each make 1296 signatures, and eight would make 1.7 million."""

    def many[
        A: (i32, u32, i64, u64, f32, f64),
        B: (i32, u32, i64, u64, f32, f64),
        C: (i32, u32, i64, u64, f32, f64),
        D: (i32, u32, i64, u64, f32, f64),
    ](a: A, b: B, c: C, d: D) -> A:
        raise NotImplementedError

    dispatched(_kept)(many)


_DRIFTS = {
    "a constraint short of the signature": (
        _constraint_short_of_the_signature, r"total: no implementations take \(f64\)"
    ),
    "parameters sharing a type parameter taking two types": (
        _parameters_that_share_a_type_parameter, r"pair: no implementations take \(i32, u8, u32\)"
    ),
    "a return other than the stub declares": (
        _return_other_than_the_stub_declares, r"does not return i32 for \(i32\)"
    ),
    "an overload of another arity": (
        _overload_of_another_arity, r"an overload takes 1 parameters, the stub 2"
    ),
    "a signature declared twice": (_signature_declared_twice, r"declares \(i32\) twice"),
    "independent type parameters past the most signatures": (
        _four_independent_type_parameters, r"declares more than 1024 signatures"
    ),
}  # fmt: skip


@pytest.mark.parametrize(("drift", "message"), _DRIFTS.values(), ids=_DRIFTS)
def test_a_dispatched_name_that_its_implementations_drift_from_is_refused_at_import(
    *, drift: Callable[[], None], message: str
) -> None:
    """Constraints, type parameters, returns, arities and the count of signatures are checked.

    An implementation's constraint covers the signatures it serves, a type parameter binds one
    type, returns and arities agree with the stub, and the signatures stay few enough to list.
    """
    with pytest.raises(TypeError, match=message):
        drift()


@gpu
def test_a_literal_that_fits_several_types_ties_and_is_refused_not_resolved_to_one() -> None:
    """`warp.sum(1)` is of i32, u32, i64 and u64 alike, and `bits.join(u8(1), 256)` fits two.

    Numba asks about the literal before its plain `int64`, and a refusal stands once given:
    `int64` would sum as `i64`, and the byte would join with its 256 cut to zero.
    """

    @kernel(threads=32)
    def summed(out: Vector[i64]) -> None:
        out[thread_index()] = warp.sum(1)

    @kernel
    def joined(out: Vector[u64]) -> None:
        out[thread_index()] = bits.join(u8(1), 256)

    every = r"\(i32\) and \(u32\) and \(i64\) and \(u64\) fit equally well"
    with pytest.raises(TypingError, match=rf"sum takes .*, not \(1\); {every}"):
        summed[32](cp.zeros(32, np.int64))
    both = r"\(u8, u8\) and \(u32, u32\) fit equally well"
    with pytest.raises(TypingError, match=rf"join takes .*, not \(u8, 256\); {both}"):
        joined[1](cp.zeros(1, np.uint64))


@gpu
def test_a_literal_is_exact_for_each_type_that_holds_it_and_the_operand_decides() -> None:
    """255 fits a u8, so `join(u8, 255)` is the byte join; 256 does not, so only `u32` holds it."""

    @kernel
    def joined(halves: Vector[u8], out: Vector[u64]) -> None:
        out[0] = bits.join(halves[0], 255)
        out[1] = bits.join(u32(halves[0]), 256)

    out = cp.zeros(2, np.uint64)
    joined[1](cp.asarray(np.array([7], np.uint8)), out)

    assert out.get().tolist() == [7 | 255 << 8, 7 | 256 << 32]
