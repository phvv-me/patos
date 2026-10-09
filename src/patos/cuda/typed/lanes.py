"""Packed lanes on the device: a type of their own over one 32-bit register.

A lanes value is no integer to Numba, so no arithmetic or implicit conversion reaches it and an
operation on it is the one its lanes pick (`primitives.bits`). It converts to and from a 32-bit
integer for free where something asks for one: a cast (`u8x4(word)`, `u32(lanes)`), a parameter, a
return or a declaration, and a store into an integer array. An integer narrower than 32 bits
drops or invents lanes, so it is refused: a source goes through `u32`.
"""

import operator
from collections.abc import Mapping, Sequence
from functools import cache

from llvmlite import ir
from numba import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.cuda.cudadecl import registry as typing_registry
from numba.cuda.cudaimpl import registry as lowering_registry
from numba.cuda.extending import models, register_model, typeof_impl

from ..scalars import Lanes


class LaneType(types.Number):
    """The device type of one kind of lanes, named as patos spells it."""

    def __init__(self, lanes: Lanes) -> None:
        super().__init__(lanes.name)
        self.lanes = lanes
        self.bitwidth = 32

    def unify(self, _context, _other: types.Type) -> None:
        """Lanes meet no other type, so a local holding them holds nothing else."""


@cache
def lane_type(lanes: Lanes) -> LaneType:
    return LaneType(lanes)


@register_model(LaneType)
class _Word(models.PrimitiveModel):
    def __init__(self, manager, lanes: LaneType) -> None:
        super().__init__(manager, lanes, ir.IntType(32))


@typeof_impl.register(Lanes)
def _typeof_lanes(lanes: Lanes, _context) -> types.NumberClass:
    return types.NumberClass(lane_type(lanes))


@lowering_registry.lower_cast(types.Integer, LaneType)
@lowering_registry.lower_cast(LaneType, types.Integer)
@lowering_registry.lower_cast(LaneType, LaneType)
def _reinterpreted(
    context, builder: ir.IRBuilder, given: types.Number, wanted: types.Number, value: ir.Value
) -> ir.Value:
    """`value` read as a 32-bit word and then as `wanted`, free between lanes and 32-bit ints.

    A 64-bit integer is the word Numba's arithmetic widened, so it truncates to it and a word
    widens to it unsigned; a narrower integer would drop or invent lanes.
    """
    if min(given.bitwidth, wanted.bitwidth) < 32:
        raise TypingError(f"{given} to {wanted}: lanes are a 32-bit word, so convert through u32")
    word = types.uint32
    if not isinstance(given, LaneType):
        value = context.cast(builder, value, given, word)
    return value if isinstance(wanted, LaneType) else context.cast(builder, value, word, wanted)


@typing_registry.register_global(operator.setitem)
class _Stored(templates.AbstractTemplate):
    """A store of lanes into an integer array converts them to its element, as a scalar's does."""

    def generic(
        self, args: Sequence[types.Type], kws: Mapping[str, types.Type]
    ) -> templates.Signature | None:
        array, index, value = args
        integers = isinstance(array, types.Array) and isinstance(array.dtype, types.Integer)
        if kws or not (isinstance(value, LaneType) and integers):
            return None
        return self.context.resolve_function_type(
            operator.setitem, (array, index, array.dtype), {}
        )
