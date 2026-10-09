"""The dtype of a device array, as the stub of `numba.cuda` lets a type checker read it."""

from typing import assert_type

import numpy as np

from patos.cuda.scalars import Numeric
from patos.cuda.typed import (
    Matrix,
    Vector,
    cuda,
    f32,
    f64,
    i16,
    i32,
    i64,
    number,
    u8,
    u16,
    u32,
    u64,
)

assert_type(cuda.shared.array(4, i32), Vector[i32])
assert_type(cuda.shared.array((4,), u64), Vector[u64])
assert_type(cuda.shared.array((4, 64), i32), Matrix[i32])
assert_type(cuda.shared.array(shape=4, dtype=u32, alignment=16), Vector[u32])
assert_type(cuda.local.array(8, i64), Vector[i64])
assert_type(cuda.local.array(8, u8), Vector[u8])
assert_type(cuda.local.array(8, i16), Vector[i16])
assert_type(cuda.local.array(8, u16), Vector[u16])
assert_type(cuda.local.array((2, 2, 2), i32), Numeric[int, int, int, int])
assert_type(cuda.local.array(8, bool), Vector[bool])
assert_type(cuda.local.array(8, np.uint64), Vector[np.uint64])
assert_type(cuda.local.array((2, 2, 2), np.float32), Numeric[np.float32, int, int, int])

# ty reads the class `float` as `float*`, which `assert_type` refuses as `float`, so the floats are
# held to the arrays they are assigned to.
shape: tuple[int, ...] = (2, 3)
anywhere: Numeric[int, *tuple[int, ...]] = cuda.local.array(shape, i32)
singles: Vector[f32] = cuda.local.array(8, f32)
doubles: Matrix[f64] = cuda.local.array((2, 2), f64)
numbers: Vector[number] = cuda.local.array(8, f32)


def _staged[T: (i32, u32, i64, u64, f32, f64)](reduced: T, partial: Vector[number]) -> None:
    partial[0] = reduced


def _total[T: (i32, u32, i64, u64, f32, f64)](value: T) -> T:
    """`type(value)` is a bare `type` to pyrefly and `type[T]` to ty; neither refuses the sum."""
    partial = cuda.shared.array(4, type(value))
    _staged(value, partial)
    summed = partial[0]
    for index in range(1, 4):
        summed = max(summed, partial[index])
    return summed


def _common[T: (u32, u64)](value: T) -> T:
    partial = cuda.shared.array(4, type(value))
    _staged(value, partial)
    bits = partial[0]
    for index in range(1, 4):
        bits = type(value)(bits & partial[index])
    return bits


assert_type(_total(1), int)
assert_type(_common(1), int)
