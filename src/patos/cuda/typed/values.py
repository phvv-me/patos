"""The device methods and attributes a named value's class defines.

Numba types every named value as one of two tuple types whatever its class, so one attribute
template answers for all of them, finding the member on the class the tuple was built from, and
each member's name lowers once, calling that class's member; a field another class names alike
stays a field.
"""

import inspect
from collections.abc import Mapping, Sequence
from functools import cache, partial

from llvmlite import ir
from numba import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.cuda.core import imputils
from numba.cuda.cudadecl import registry as typing_registry
from numba.cuda.cudaimpl import registry as lowering_registry
from numba.cuda.dispatcher import CUDADispatcher

from .overloads import lowered_call, typed_call

# Each named value class's device members by name: the compiled function, and whether it is read
# as an attribute rather than called.
_MEMBERS: dict[tuple[type, str], tuple[CUDADispatcher, bool]] = {}


def register_value_member(
    cls: type[tuple], name: str, device: CUDADispatcher, *, attribute: bool
) -> None:
    """Make `device` the device method `name` of `cls`'s named values, or their attribute."""
    if name.startswith("__"):
        raise TypeError(
            f"{cls.__name__}.{name}: a named value keeps the operators of a tuple; name a method"
        )
    if not any(held == name for _, held in _MEMBERS):
        _lower(name)
    _MEMBERS[cls, name] = device, attribute


@typing_registry.register_attr
class _Members(templates.AttributeTemplate):
    key = types.BaseNamedTuple

    def generic_resolve(self, value: types.BaseNamedTuple, attr: str) -> types.Type | None:
        held = _MEMBERS.get((value.instance_class, attr))
        if held is None:
            return None
        device, attribute = held
        if attribute:
            return typed_call(self.context, device, (value,)).return_type
        return types.BoundFunction(_method(attr), value)


@cache
def _method(attr: str) -> type[templates.AbstractTemplate]:
    """The template typing a call of the method `attr` on the named value it is bound to."""

    class Method(templates.AbstractTemplate):
        key = (_Members, attr)

        def generic(
            self, args: Sequence[types.Type], kws: Mapping[str, types.Type]
        ) -> templates.Signature:
            device, _ = _MEMBERS[self.this.instance_class, attr]
            try:
                inspect.signature(device.py_func).bind(self.this, *args)
            except TypeError as error:
                raise TypingError(f"{self.this.instance_class.__name__}.{attr}: {error}") from None
            if kws:
                raise TypingError(f"{attr} takes its arguments by position")
            return typed_call(self.context, device, (self.this, *args)).as_method()

    return Method


def _lower(attr: str) -> None:
    """Lower a call of the method `attr` and a read of the attribute `attr`, whatever the class."""
    method = (_Members, attr)
    lowering_registry.lower(method, types.BaseNamedTuple, types.VarArg(types.Any))(
        partial(_called, attr)
    )
    lowering_registry.lower_getattr(types.BaseNamedTuple, attr)(partial(_read, attr))


def _called(
    attr: str, context, builder: ir.IRBuilder, call: templates.Signature, values: Sequence
) -> ir.Value:
    """Call the member `attr` of the named value class `call`'s first argument was built from."""
    device, _ = _MEMBERS[call.args[0].instance_class, attr]
    return lowered_call(
        device, context, builder, typed_call(context.typing_context, device, call.args), values
    )


def _read(
    attr: str, context, builder: ir.IRBuilder, value_type: types.BaseNamedTuple, value: ir.Value
) -> ir.Value:
    """Read the attribute `attr`, or the field another class names alike."""
    if attr not in value_type.fields:
        return _called(attr, context, builder, templates.signature(None, value_type), [value])
    index = value_type.fields.index(attr)
    field = builder.extract_value(value, index)
    return imputils.impl_ret_borrowed(context, builder, value_type[index], field)
