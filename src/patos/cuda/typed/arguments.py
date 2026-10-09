"""What a kernel receives for each value it is launched with, a record, a device array or a
scalar: the Numba type it compiles at and the flat values that type lays out."""

import math
from functools import cache
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import cupy as cp
import numpy as np
from numba import types
from numba.cuda.np.numpy_support import from_dtype

from .records import RecordType, record_type

if TYPE_CHECKING:
    from ..scalars import Shaped
    from .struct import Struct

# What a kernel receives for one value: its Numba type and the values that type flattens to.
type Argument = tuple[types.Type, tuple]


def argument(value: Struct | Shaped | np.generic | int | float | bool) -> Argument:
    """What a kernel receives for `value`.

    A record keeps the argument it marshalled to, an array's is its descriptor, and a scalar
    passes as it is, a Python one as Numba's default type of it.
    """
    cls = type(value)
    scalar = _SCALARS.get(cls)
    if scalar is not None:
        return scalar, (value,)
    marshalled = getattr(cls, "__kernel_argument__", None)
    if marshalled is None:
        return _array(cast("cp.ndarray | _Interfaced", value))
    return marshalled(value)


def record_argument(
    cls: type[Struct], fields: dict[str, Argument], constants: tuple[tuple[str, int], ...]
) -> Argument:
    """The argument of a `cls` record whose fields marshalled to `fields`, in order, and whose
    constants, part of its type, marshal to nothing."""
    kinds: list[types.Type] = []
    values: list = []
    for kind, held in fields.values():
        kinds.append(kind)
        values.extend(held)
    # Numba interns its types and every record type cached here holds the types of its fields,
    # so they live as long as the key that names them by identity, which costs no hashing.
    key = (cls, constants, *map(id, kinds))
    kind = _records.get(key)
    if kind is None:
        kind = _records[key] = record_type(cls)(tuple(zip(fields, kinds, strict=True)), constants)
    return kind, tuple(values)


_records: dict[tuple, RecordType] = {}
_SCALARS: dict[type, types.Type] = {
    bool: types.boolean, int: types.int64, float: types.float64,
    **{kind: from_dtype(np.dtype(kind)) for kind in (
        np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
        np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128,
    )},
}  # fmt: skip


class _Interface(TypedDict):
    typestr: str
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    data: tuple[int, bool]


class _Interfaced(Protocol):
    """A device array of any module, known by its CUDA array interface."""

    @property
    def __cuda_array_interface__(self) -> _Interface: ...


def _array(value: cp.ndarray | _Interfaced) -> Argument:
    """A device array's descriptor as Numba's array model lays it out.

    A CuPy array is read by attribute, any other one from its interface.
    """
    if isinstance(value, cp.ndarray):
        dtype, shape, strides, pointer = value.dtype, value.shape, value.strides, value.data.ptr
    else:
        interface = value.__cuda_array_interface__
        dtype, shape = np.dtype(interface["typestr"]), interface["shape"]
        strides, pointer = interface["strides"], interface["data"][0]
    itemsize = dtype.itemsize
    if len(shape) == 1:
        strides = strides or (itemsize,)
        kind = _array_type(dtype, 1, shape[0] <= 1 or strides[0] == itemsize)
        return kind, (0, 0, shape[0], itemsize, pointer, shape[0], strides[0])
    contiguous = _contiguous_strides(shape, itemsize)
    strides = strides or contiguous
    # As numpy's flags read it, a dimension of one element strides nowhere and an empty array
    # is contiguous whatever its strides.
    laid = 0 in shape or all(
        extent <= 1 or stride == step
        for extent, stride, step in zip(shape, strides, contiguous, strict=True)
    )
    kind = _array_type(dtype, len(shape), laid)
    return kind, (0, 0, math.prod(shape), itemsize, pointer, *shape, *strides)


@cache
def _array_type(dtype: np.dtype, ndim: int, contiguous: bool) -> types.Array:
    return types.Array(from_dtype(dtype), ndim, "C" if contiguous else "A")


def _contiguous_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    strides, stride = [], itemsize
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))
