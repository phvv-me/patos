"""A `Monoid` over a whole device array, the level above the block's.

`reduce`, `scan` and `exclusive` take the monoids the warp and block engines take, and run on
cuda.compute (CUB), each launch on the `stream` the caller names. The built-in monoids map onto its
well-known operators, which need no JIT. A monoid of any other `combine` is refused: cuda.compute
compiles a plain Python function through its own JIT, and neither a numba-cuda device function nor
a patos one (`cuda-cccl` 1.2.1, `numba-cuda` 0.30.4) passes through it.

Each algorithm is built once per operator and dtype, which costs about a second, and then runs in
temporary storage kept per stream: a call is a launch, where a one-shot call builds again.

An array scans in place by naming itself as the output.
"""

from collections.abc import Callable
from functools import cache

import cupy as cp
import numpy as np
from cuda.compute import (
    OpKind,
    make_exclusive_scan,
    make_inclusive_scan,
    make_reduce_into,
)

from ..runtime.streams import StreamLike, handle
from ..scalars import Shaped
from . import monoid


def _highest(dtype: np.dtype) -> np.generic:
    return dtype.type(np.inf if dtype.kind == "f" else np.iinfo(dtype).max)


def _lowest(dtype: np.dtype) -> np.generic:
    return dtype.type(-np.inf if dtype.kind == "f" else np.iinfo(dtype).min)


# Each built-in monoid's cuda.compute operator and its identity at a dtype.
_NATIVE: dict[monoid.Monoid, tuple[OpKind, Callable[[np.dtype], np.generic]]] = {
    monoid.add: (OpKind.PLUS, lambda dtype: dtype.type(0)),
    monoid.minimum: (OpKind.MINIMUM, _highest),
    monoid.maximum: (OpKind.MAXIMUM, _lowest),
    monoid.bitwise_and: (OpKind.BIT_AND, lambda dtype: np.invert(dtype.type(0))),
    monoid.bitwise_or: (OpKind.BIT_OR, lambda dtype: dtype.type(0)),
    monoid.bitwise_xor: (OpKind.BIT_XOR, lambda dtype: dtype.type(0)),
}
# Each algorithm's builder, and what its call names the value it starts from.
_ALGORITHMS = {
    "reduce": (make_reduce_into, "h_init"),
    "scan": (make_inclusive_scan, "init_value"),
    "exclusive": (make_exclusive_scan, "init_value"),
}


def _native(operator: monoid.Monoid) -> tuple[OpKind, Callable[[np.dtype], np.generic]]:
    try:
        return _NATIVE[operator]
    except KeyError:
        raise TypeError(
            "a whole array reduces and scans by the built-in monoids only: cuda.compute takes no "
            "numba-cuda device function"
        ) from None


def identity(operator: monoid.Monoid, dtype: np.dtype) -> np.generic:
    """The value of `dtype` that the built-in `operator` leaves any other as it is."""
    return _native(operator)[1](dtype)


class _Built:
    """One algorithm built for an operator and dtype, and the storage it runs in on each stream."""

    def __call__[A: Shaped](
        self, values: A, out: A, *, kind: OpKind, first: np.ndarray, stream: StreamLike
    ) -> None:
        """Run over `values` into `out` from `first`, in storage grown to the largest size seen."""
        items = int(values.shape[0])
        arguments = {"d_in": values, "d_out": out, "op": kind, self.start: first, "stream": stream}
        number = handle(stream)
        held, storage = self.temporary.get(number, (0, None))
        if items > held:
            need = self.run(temp_storage=None, num_items=items, **arguments)
            storage = cp.empty(max(int(need), 1), np.uint8)
            self.temporary[number] = items, storage
        self.run(temp_storage=storage, num_items=items, **arguments)

    def __init__(self, name: str, kind: OpKind, dtype: np.dtype) -> None:
        make, self.start = _ALGORITHMS[name]
        example = cp.empty(1, dtype)
        self.run = make(d_in=example, d_out=example, op=kind, **{self.start: np.zeros(1, dtype)})
        self.temporary: dict[int, tuple[int, cp.ndarray]] = {}


@cache
def _built(name: str, kind: OpKind, dtype: np.dtype) -> _Built:
    return _Built(name, kind, dtype)


def _run[A: Shaped](
    name: str, operator: monoid.Monoid, values: A, *, out: A, stream: StreamLike
) -> A:
    kind, start = _native(operator)
    _built(name, kind, values.dtype)(
        values, out, kind=kind, first=np.array([start(values.dtype)]), stream=stream
    )
    return out


def reduce[A: Shaped](operator: monoid.Monoid, values: A, *, out: A, stream: StreamLike) -> A:
    """Reduce `values` by `operator` into the one element of `out` on `stream`, returning `out`."""
    return _run("reduce", operator, values, out=out, stream=stream)


def scan[A: Shaped](operator: monoid.Monoid, values: A, *, out: A, stream: StreamLike) -> A:
    """Write into `out` the combination of `values` up to each, itself included, and return it."""
    return _run("scan", operator, values, out=out, stream=stream)


def exclusive[A: Shaped](operator: monoid.Monoid, values: A, *, out: A, stream: StreamLike) -> A:
    """Write into `out` the combination of `values` below each, the first the identity."""
    return _run("exclusive", operator, values, out=out, stream=stream)
