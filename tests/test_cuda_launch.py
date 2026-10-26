"""A launch runs on the stream it is given, of whichever library, and on no other: it names one,
and an array whose interface names another stream as its producer is waited for."""

import cupy as cp
import numpy as np
import pytest
from cuda.core import Device
from patos.cuda.runtime import ArrayInterface
from patos.cuda.typed import Vector, i32, items, kernel

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")


@kernel
def counted(out: Vector[i32]) -> None:
    for item in items(out.size):
        out[item] += item


@kernel
def copied(source: Vector[i32], out: Vector[i32]) -> None:
    for item in items(out.size):
        out[item] = source[item]


class Produced:
    """A CuPy array whose interface names `stream` as the stream that produced it."""

    @property
    def __cuda_array_interface__(self) -> ArrayInterface:
        interface = self.array.__cuda_array_interface__
        interface["stream"] = self.stream.ptr
        return interface

    def __init__(self, array: cp.ndarray, stream: cp.cuda.Stream) -> None:
        """array: an `int32` array with shape `[n]`. stream: the stream that wrote it."""
        self.array = array
        self.stream = stream


def test_a_launch_runs_on_the_stream_it_is_given() -> None:
    """A CuPy stream and a cuda.core stream each run the launch as their own work."""
    out = cp.zeros(64, np.int32)
    cupy_stream = cp.cuda.Stream(non_blocking=True)
    core_stream = Device().create_stream()
    cp.cuda.Device().synchronize()

    counted[64](out, stream=cupy_stream)
    cupy_stream.synchronize()
    counted[64](out, stream=core_stream)
    core_stream.sync()

    assert out.get().tolist() == [2 * item for item in range(64)]


def test_a_launch_names_its_stream() -> None:
    with pytest.raises(TypeError, match="stream"):
        counted[64](cp.zeros(64, np.int32))


def test_a_launch_waits_for_the_stream_an_array_names_as_its_producer() -> None:
    """The producer is busy before it writes, so a launch that did not wait would copy zeros."""
    producer, consumer = (cp.cuda.Stream(non_blocking=True) for _ in range(2))
    source, out = cp.zeros(1 << 20, np.int32), cp.zeros(1 << 20, np.int32)
    busy = cp.ones(1 << 22, np.int32)
    cp.cuda.Device().synchronize()
    with producer:
        for _ in range(200):
            cp.cumsum(busy, out=busy)
        source.fill(7)

    copied[source.size](Produced(source, producer), out, stream=consumer)
    consumer.synchronize()

    assert bool((out == 7).all())
