"""The whole-array engine at every built-in monoid, each against a host scan."""

import cupy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from patos.cuda.primitives import bits, monoid, wide

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

_UFUNCS = {
    name: getattr(np, name)
    for name in ("add", "minimum", "maximum", "bitwise_and", "bitwise_or", "bitwise_xor")
}
_NUMBERS = [np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64]
_CASES = [
    (name, kind)
    for name in _UFUNCS
    for kind in _NUMBERS
    if name in ("add", "minimum", "maximum") or np.issubdtype(kind, np.integer)
]


def _values(kind: type[np.number]) -> st.SearchStrategy[int | float]:
    """Integers over the whole range of an integer type, whole floats so a sum is exact."""
    if not issubclass(kind, np.integer):
        return st.integers(-(2**10), 2**10).map(float)
    info = np.iinfo(kind)
    return st.one_of(st.integers(info.min, info.max), st.sampled_from([info.min, info.max, 0]))


@pytest.mark.parametrize(("name", "kind"), _CASES)
@settings(deadline=None, max_examples=10)
@given(data=st.data())
def test_a_whole_array_reduces_and_scans_as_the_host_does(
    *, name: str, kind: type[np.number], data: st.DataObject
) -> None:
    """The reduction, the inclusive scan and the exclusive scan of a drawn array match numpy's.

    The exclusive scan opens with the monoid's identity, and the inclusive and the exclusive scan
    in place are the scans into another array.
    """
    size = data.draw(st.integers(1, 3000))
    host = np.array(data.draw(st.lists(_values(kind), min_size=size, max_size=size)), kind)
    operator, stream = getattr(monoid, name), cp.cuda.Stream.null
    inclusive = _UFUNCS[name].accumulate(host, dtype=kind)
    exclusive = np.concatenate([[wide.identity(operator, np.dtype(kind))], inclusive[:-1]])
    results = {}
    for scan in (wide.scan, wide.exclusive):
        copied, in_place = cp.asarray(host), cp.asarray(host)
        results[scan] = (
            scan(operator, copied, out=cp.empty_like(copied), stream=stream).get(),
            scan(operator, in_place, out=in_place, stream=stream).get(),
        )
    total = wide.reduce(operator, cp.asarray(host), out=cp.zeros(1, kind), stream=stream).get()

    assert results[wide.scan][0].tolist() == results[wide.scan][1].tolist() == inclusive.tolist()
    assert results[wide.exclusive][0].tolist() == results[wide.exclusive][1].tolist()
    assert results[wide.exclusive][0].tolist() == exclusive.tolist()
    assert total.tolist() == [inclusive[-1]]


def test_a_monoid_cuda_compute_has_no_operator_for_is_refused() -> None:
    """cuda.compute takes no numba-cuda device function, so a custom monoid has no array run."""
    values = cp.zeros(4, np.uint32)
    with pytest.raises(TypeError, match="built-in monoids only"):
        wide.scan(bits.bytes_min, values, out=values, stream=cp.cuda.Stream.null)
