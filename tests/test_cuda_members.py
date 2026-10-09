"""The device methods and attributes of a record: a call of the device function itself."""

import cupy as cp
import numpy as np
import pytest
from numba.cuda import dispatcher

from patos.cuda.typed import Struct, Vector, device, i32, i64, items, kernel, kernelcache, u64

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")


class Held(Struct):
    """A table read through a method, a property and an operator."""

    slots: Vector[u64]

    @device
    def __getitem__(self, index: i32) -> u64:
        return self.slots[index]

    @property
    @device
    def total(self) -> i64:
        return len(self.slots)

    @device
    def shifted(self, by: i32) -> u64:
        return self.slots[0] >> by


@kernel
def reading(held: Held, out: Vector[u64]) -> None:
    for item in items(out.size):
        out[item] = held.shifted(by=1) + held.total + held[item]


@pytest.fixture
def compiled(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The names of the functions Numba compiles, with no cached kernel to hide a compile."""
    names: list[str] = []
    original = dispatcher.compile_cuda

    def spy(pyfunc, *arguments, **options):
        names.append(pyfunc.__name__)
        return original(pyfunc, *arguments, **options)

    monkeypatch.setattr(kernelcache.Persistent, "enable_caching", lambda _: None)
    monkeypatch.setattr(dispatcher, "compile_cuda", spy)
    return names


def test_a_record_method_and_property_are_called_with_no_function_made_between(
    compiled: list[str],
) -> None:
    """Only an operator is answered through an overload, a forwarder compiled per call shape.

    The method and the property lower as the call of their own device function, as a named
    value's do, so they cost no module of their own.
    """
    slots = cp.asarray(np.array([8, 5, 3], np.uint64))
    out = cp.zeros(3, np.uint64)
    reading[3](Held(slots), out)

    assert out.get().tolist() == [4 + 3 + 8, 4 + 3 + 5, 4 + 3 + 3]
    assert {"shifted", "total", "__getitem__"} <= set(compiled)
    assert sum(name == "_patos_forwarded" for name in compiled) == 1
