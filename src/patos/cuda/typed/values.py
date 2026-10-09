"""The device methods and attributes a named value's class defines.

Numba types every named value as one of two tuple types whatever its class, so one attribute
template answers for all of them (`members`), finding the member on the class the tuple was built
from; a field another class names alike stays a field.
"""

from llvmlite import ir
from numba import types
from numba.core.typing import templates
from numba.cuda.core import imputils
from numba.cuda.cudadecl import registry as typing_registry
from numba.cuda.dispatcher import CUDADispatcher

from .members import Members


def _tuple_field(
    context, builder: ir.IRBuilder, value_type: types.BaseNamedTuple, value: ir.Value, attr: str
) -> ir.Value:
    index = value_type.fields.index(attr)
    field = builder.extract_value(value, index)
    return imputils.impl_ret_borrowed(context, builder, value_type[index], field)


# A named value's class is what Numba's tuple type remembers it was made from.
_VALUES = Members(types.BaseNamedTuple, lambda value: value.instance_class, _tuple_field)


@typing_registry.register_attr
class _Named(templates.AttributeTemplate):
    key = types.BaseNamedTuple

    def generic_resolve(self, value: types.BaseNamedTuple, attr: str) -> types.Type | None:
        return _VALUES.resolved(self.context, value, attr)


def register_value_member(
    cls: type[tuple], name: str, device: CUDADispatcher, *, attribute: bool
) -> None:
    """Make `device` the device method `name` of `cls`'s named values, or their attribute."""
    if name.startswith("__"):
        raise TypeError(
            f"{cls.__name__}.{name}: a named value keeps the operators of a tuple; name a method"
        )
    _VALUES.register(cls, name, device, attribute=attribute)
