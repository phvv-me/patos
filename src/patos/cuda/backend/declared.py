"""What a declaration of `patos.cuda.front` is to Numba: the type it names, and the types it
admits."""

import numpy as np
from numba import types
from numba.cuda.np import numpy_support

from ..front.declarations import Declared, NamedValue, Returns, is_scalar
from ..scalars import ArrayOf, Lanes
from .lanes import lane_type


def numba_type(declared: Returns) -> types.Type | None:
    """The Numba type a scalar, `bool`, lanes or a tuple of these declares, else None."""
    if isinstance(declared, Lanes):
        return lane_type(declared)
    if isinstance(declared, tuple):
        elements = [numba_type(element) for element in declared]
        return None if None in elements else types.BaseTuple.from_types(elements)
    if is_scalar(declared):
        return getattr(types, np.dtype(declared).name)
    return types.boolean if declared is bool else None


def is_admitted(kind: Declared, given: types.Type) -> bool:
    """Whether `given` is a value of what `kind` declares.

    An array admits its element and dimension count, and a named value each field in turn.
    """
    if isinstance(kind, ArrayOf):
        return isinstance(given, types.Array) and kind.admits(
            numpy_support.as_dtype(given.dtype), given.ndim
        )
    if isinstance(kind, NamedValue):
        return (
            isinstance(given, types.BaseNamedTuple)
            and given.instance_class is kind.cls
            and len(given) == len(kind.fields)
            and all(
                is_admitted(field, held) for field, held in zip(kind.kinds(), given, strict=True)
            )
        )
    expected = numba_type(kind)
    return expected is None or types.unliteral(given) == expected
