"""The device methods and attributes a class defines for its values, a named value's or a record's.

Numba types every named value as one of two tuple types whatever its class, and a record as a type
of its own. One attribute template answers for each family, finding the member on the class the
value was built from, and each member's name lowers once, as a call of that class's device
function itself, with no function made between. A field another class names alike stays a field.
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from functools import partial

from llvmlite import ir
from numba import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.cuda.cudaimpl import registry as lowering_registry
from numba.cuda.dispatcher import CUDADispatcher

from .overloads import lowered_call, typed_call


class Members:
    """The device members of the classes whose values are one family of Numba types.

    family: the Numba type of such a value, or the base of the types.
    owner: the class a value of that type was built from.
    field: reads field `attr` of a value, which a member of another class may share its name with.
    """

    def __init__(
        self,
        family: type[types.Type],
        owner: Callable[[types.Type], type],
        field: Callable[..., ir.Value],
    ) -> None:
        self.family = family
        self.owner = owner
        self.field = field
        self.held: dict[tuple[type, str], tuple[CUDADispatcher, bool]] = {}
        self.methods: dict[str, type[templates.AbstractTemplate]] = {}

    def register(self, cls: type, name: str, device: CUDADispatcher, *, attribute: bool) -> None:
        """Make `device` the device method `name` of `cls`'s values, or their attribute."""
        if name not in self.methods:
            self._lower(name)
        self.held[cls, name] = device, attribute

    def resolved(self, context, value: types.Type, attr: str) -> types.Type | None:
        """The type of member `attr` of `value`, None where its class defines no such member."""
        held = self.held.get((self.owner(value), attr))
        if held is None:
            return None
        device, attribute = held
        if attribute:
            return typed_call(context, device, (value,)).return_type
        return types.BoundFunction(self.methods[attr], value)

    def _called(
        self,
        attr: str,
        context,
        builder: ir.IRBuilder,
        call: templates.Signature,
        values: Sequence,
    ) -> ir.Value:
        """Call the member `attr` of the class `call`'s first argument was built from."""
        device, _ = self.held[self.owner(call.args[0]), attr]
        typed = typed_call(context.typing_context, device, call.args)
        return lowered_call(device, context, builder, typed, values)

    def _lower(self, attr: str) -> None:
        """Lower a call of the method `attr` and a read of the attribute `attr` for any class."""
        members = self

        class Method(templates.AbstractTemplate):
            key = (members, attr)

            def generic(
                self, args: Sequence[types.Type], kws: Mapping[str, types.Type]
            ) -> templates.Signature:
                cls = members.owner(self.this)
                device, _ = members.held[cls, attr]
                try:
                    bound = inspect.signature(device.py_func).bind(self.this, *args, **kws)
                except TypeError as error:
                    raise TypingError(f"{cls.__name__}.{attr}: {error}") from None
                return typed_call(self.context, device, bound.args).as_method()

        self.methods[attr] = Method
        lowering_registry.lower((self, attr), self.family, types.VarArg(types.Any))(
            partial(self._called, attr)
        )
        lowering_registry.lower_getattr(self.family, attr)(partial(self._read, attr))

    def _read(
        self, attr: str, context, builder: ir.IRBuilder, value_type: types.Type, value: ir.Value
    ) -> ir.Value:
        """Read the attribute `attr`, or the field another class of the family names alike."""
        if (self.owner(value_type), attr) not in self.held:
            return self.field(context, builder, value_type, value, attr)
        device, _ = self.held[self.owner(value_type), attr]
        typed = typed_call(context.typing_context, device, (value_type,))
        return lowered_call(device, context, builder, typed, [value])
