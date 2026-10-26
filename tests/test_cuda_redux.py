"""Every `redux` strategy against the butterfly it replaces, and the NaNs it reduces."""

from functools import cache

import cupy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from patos.cuda.primitives import monoid, redux, warp
from patos.cuda.typed import Kernel, Matrix, Per, kernel, number, thread_index

_STREAM = cp.cuda.Stream.null

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

_THREADS = 128
_EXTREMES = ("minimum", "maximum")
# Each (monoid, dtype) a strategy serves, as the table names it.
_CASES = [(name, np.dtype(str(kind))) for name, kind in redux.served()]


@cache
def reducing(name: str) -> Kernel:
    """A kernel giving every thread the reduction of its warp, by `reduce` and by the butterfly."""
    operator = getattr(monoid, name)

    @kernel(per=Per.BLOCK, threads=_THREADS)
    def reduced(table: Matrix[number]) -> None:
        thread = thread_index()
        table[1, thread] = warp.reduce(operator, table[0, thread])
        table[2, thread] = warp.butterfly(operator, table[0, thread])

    return reduced


def _values(dtype: np.dtype) -> st.SearchStrategy[int | float]:
    """Values over the whole range of an integer type, or finite floats with the zeros and ends."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return st.one_of(
            st.integers(info.min, info.max), st.sampled_from([info.min, info.max, 0, 1])
        )
    limit = float(np.finfo(dtype).max)
    return st.one_of(
        st.floats(-limit, limit, width=32 if dtype == np.float32 else 64, allow_nan=False),
        st.sampled_from([0.0, -0.0, limit, -limit]),
    )


@pytest.mark.parametrize(("name", "dtype"), _CASES)
@settings(deadline=None, max_examples=15)
@given(data=st.data())
def test_a_strategy_reduces_every_warp_as_the_butterfly_does(
    *, name: str, dtype: np.dtype, data: st.DataObject
) -> None:
    """The strategy and the butterfly agree on every warp, a sum wrapping and a zero either sign.

    The extremes of the type sit among the values, a few warps hold one value alone, and a float
    compares by value, so that `-0.0` and `0.0` are one.
    """
    count = data.draw(st.integers(1, 3)) * _THREADS
    values = np.array(data.draw(st.lists(_values(dtype), min_size=count, max_size=count)), dtype)
    table = cp.asarray(np.concatenate([values[None], np.zeros((2, count), dtype)]))
    reducing(name)[count // _THREADS](table, stream=_STREAM)
    got = table.get()
    assert got[1].tolist() == got[2].tolist()


def test_a_float_extreme_passes_over_a_nan_and_is_nan_where_every_lane_is() -> None:
    """A NaN is a lane with nothing to give, and `-0.0` is below `0.0`.

    Warp 0 holds only NaN, warp 1 NaN among numbers, warp 2 both zeros.
    """
    warps = [
        np.full(32, np.nan),
        np.tile([np.nan, 3.0, -2.0, 0.0], 8),
        np.tile([0.0, -0.0], 16),
        np.zeros(32),
    ]
    values = np.concatenate(warps).astype(np.float32)
    tables = {name: cp.asarray(np.stack([values, 0 * values, 0 * values])) for name in _EXTREMES}
    for name, table in tables.items():
        reducing(name)[1](table, stream=_STREAM)
    low, high = (tables[name].get()[1][::32] for name in _EXTREMES)
    assert np.isnan(low[0]) and np.isnan(high[0])
    assert (low[1], high[1]) == (-2.0, 3.0)
    assert np.signbit(low[2]) and not np.signbit(high[2])
