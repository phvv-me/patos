"""The warp and block engines at every monoid they take, each against a host scan."""

from collections.abc import Callable
from functools import cache

import cupy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from patos.cuda.primitives import bits, block, monoid, warp
from patos.cuda.typed import Kernel, Matrix, Per, kernel, number, thread_index

_STREAM = cp.cuda.Stream.null

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

_THREADS = 128
_WARP = 32


def _accumulating(ufunc: np.ufunc) -> Callable[[np.ndarray], np.ndarray]:
    """The inclusive scan of `ufunc` at the type of the values, a sum wrapping as `T` does."""
    return lambda group: ufunc.accumulate(group, dtype=group.dtype)


def _lanewise(ufunc: np.ufunc) -> Callable[[np.ndarray], np.ndarray]:
    """The inclusive scan of `ufunc` over the four bytes of each `u32` word, byte by byte."""

    def _scan(words: np.ndarray) -> np.ndarray:
        return ufunc.accumulate(words.view(np.uint8).reshape(-1, 4)).view(np.uint32)[:, 0]

    return _scan


# Each monoid by name, with its host scan and which end of its type its identity is: the highest
# (True), the lowest (False) or zero (None).
_MONOIDS: dict[str, tuple[monoid.Monoid, Callable[[np.ndarray], np.ndarray], bool | None]] = {
    "add": (monoid.add, _accumulating(np.add), None),
    "minimum": (monoid.minimum, _accumulating(np.minimum), True),
    "maximum": (monoid.maximum, _accumulating(np.maximum), False),
    "bitwise_and": (monoid.bitwise_and, _accumulating(np.bitwise_and), True),
    "bitwise_or": (monoid.bitwise_or, _accumulating(np.bitwise_or), None),
    "bitwise_xor": (monoid.bitwise_xor, _accumulating(np.bitwise_xor), None),
    "bytes_min": (bits.bytes_min, _lanewise(np.minimum), True),
    "bytes_max": (bits.bytes_max, _lanewise(np.maximum), None),
}  # fmt: skip
# Each monoid over each type it takes: any number, an unsigned word for the bitwise ones.
_CASES = [
    ("add", np.int32), ("add", np.uint32), ("add", np.int64), ("add", np.uint64),
    ("add", np.float32), ("add", np.float64), ("minimum", np.int32), ("minimum", np.uint32),
    ("minimum", np.int64), ("minimum", np.uint64), ("minimum", np.float32),
    ("minimum", np.float64), ("maximum", np.int32), ("maximum", np.uint32),
    ("maximum", np.int64), ("maximum", np.uint64), ("maximum", np.float32),
    ("maximum", np.float64), ("bitwise_and", np.uint32), ("bitwise_and", np.uint64),
    ("bitwise_or", np.uint32), ("bitwise_or", np.uint64), ("bitwise_xor", np.uint32),
    ("bitwise_xor", np.uint64), ("bytes_min", np.uint32), ("bytes_max", np.uint32),
]  # fmt: skip


@cache
def folding(name: str) -> Kernel:
    """A kernel giving every thread what the monoid `name` makes of its warp and its block.

    Row 0 of the table holds the values. Rows 1 to 3 get the warp's reduction, inclusive scan and
    exclusive scan, rows 4 to 7 the block's reduction, inclusive scan, exclusive scan and total.
    """
    operator, across = _MONOIDS[name][0], block.over(_THREADS)

    @kernel(per=Per.BLOCK, threads=_THREADS)
    def fold(table: Matrix[number]) -> None:
        thread = thread_index()
        value = table[0, thread]
        table[1, thread] = warp.reduce(operator, value)
        table[2, thread] = warp.scan(operator, value)
        table[3, thread] = warp.exclusive(operator, value)
        table[4, thread] = across.reduce(operator, value)
        table[5, thread] = across.scan(operator, value)
        before, total = across.exclusive(operator, value)
        table[6, thread] = before
        table[7, thread] = total

    return fold


def _extreme(kind: type[np.number], *, high: bool) -> np.number:
    """The highest or the lowest value of `kind`, an infinity for a float."""
    if issubclass(kind, np.integer):
        return kind(np.iinfo(kind).max if high else np.iinfo(kind).min)
    if issubclass(kind, np.floating):
        return kind(np.inf if high else -np.inf)
    raise TypeError(kind)


def _identity(name: str, kind: type[np.number]) -> np.number:
    """What the monoid `name` leaves any value of `kind` as."""
    high = _MONOIDS[name][2]
    return kind(0) if high is None else _extreme(kind, high=high)


def _folded(name: str, kind: type[np.number], values: np.ndarray) -> np.ndarray:
    """What each thread of every warp and block of `values` gets, as the rows of the table.

    values: `kind` of shape `[n]`, a multiple of the block.
    """
    rows = np.zeros((7, values.size), kind)
    for width, first in ((_WARP, 0), (_THREADS, 3)):
        for start in range(0, values.size, width):
            inclusive = _MONOIDS[name][1](values[start : start + width])
            exclusive = np.concatenate([[_identity(name, kind)], inclusive[:-1]]).astype(kind)
            stops = (np.full(width, inclusive[-1], kind), inclusive, exclusive)
            rows[first : first + 3, start : start + width] = stops
            if width == _THREADS:
                rows[6, start : start + width] = inclusive[-1]
    return rows


def _values_of(kind: type[np.number]) -> st.SearchStrategy[int | float]:
    """Values of `kind`, integers over the whole range and floats whole, so a sum is exact."""
    if not issubclass(kind, np.integer):
        return st.integers(-(2**12), 2**12).map(float)
    info = np.iinfo(kind)
    return st.one_of(st.integers(info.min, info.max), st.sampled_from([info.min, info.max, 0, 1]))


@pytest.mark.parametrize(("name", "kind"), _CASES)
@settings(deadline=None, max_examples=10)
@given(data=st.data())
def test_a_monoid_reduces_and_scans_each_warp_and_block_as_the_host_does(
    *, name: str, kind: type[np.number], data: st.DataObject
) -> None:
    """Every thread gets the reduction, the inclusive and the exclusive scan of its warp and block.

    The exclusive scan opens with the identity, and a block's total follows its last thread.
    """
    count = data.draw(st.integers(1, 3)) * _THREADS
    values = np.array(data.draw(st.lists(_values_of(kind), min_size=count, max_size=count)), kind)
    table = cp.asarray(np.concatenate([values[None], np.zeros((7, count), kind)]))
    folding(name)[count // _THREADS](table, stream=_STREAM)
    assert table[1:].get().tolist() == _folded(name, kind, values).tolist()
