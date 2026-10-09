"""A record on the device: its own Numba type, a C struct of its fields, and the device methods,
operators and properties its class declares.

A record is no tuple to Numba, so `len`, indexing and `in` mean only what its class declares.
"""

import inspect
import operator
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, ClassVar

from llvmlite import ir
from numba import types
from numba.cuda import cgutils
from numba.cuda.core.imputils import impl_ret_borrowed
from numba.cuda.cudadecl import registry as typing_registry
from numba.cuda.cudaimpl import registry as lowering_registry
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda.extending import models, overload, register_model
from numba.cuda.typing.templates import AttributeTemplate

from .members import Members

if TYPE_CHECKING:
    from .struct import Struct

# The host protocol a device operator answers, where `operator` names it differently.
_BUILTINS: dict[str, Callable] = {"__len__": len, "__abs__": abs}


class RecordType(types.Type):
    """The device type of one record class over the types its fields hold.

    Each record class has its own subclass, so a device operator is registered on it.
    """

    cls: ClassVar[type[Struct]]

    def __init__(
        self, fields: tuple[tuple[str, types.Type], ...], constants: tuple[tuple[str, int], ...]
    ) -> None:
        self.fields = fields
        self.members = dict(fields)
        self.constants = dict(constants)
        held = [f"{name}: {kind}" for name, kind in fields]
        held += [f"{name}={value!r}" for name, value in constants]
        super().__init__(f"{self.cls.__module__}.{self.cls.__qualname__}({', '.join(held)})")


class _Model(models.StructModel):
    def __init__(self, manager, record: RecordType) -> None:
        super().__init__(manager, record, list(record.fields))


@typing_registry.register_attr
class _Fields(AttributeTemplate):
    key = RecordType

    def generic_resolve(self, record: RecordType, attr: str) -> types.Type | None:
        if (member := _RECORDS.resolved(self.context, record, attr)) is not None:
            return member
        if attr in record.constants:
            return types.literal(record.constants[attr])
        return record.members.get(attr)


@lowering_registry.lower_getattr_generic(RecordType)
def _field(context, builder: ir.IRBuilder, record: RecordType, value: ir.Value, attr: str):
    if attr in record.constants:
        constant = record.constants[attr]
        return context.get_constant(types.literal(constant).literal_type, constant)
    fields = cgutils.create_struct_proxy(record)(context, builder, value=value)
    return impl_ret_borrowed(context, builder, record.members[attr], getattr(fields, attr))


# A record's class is the one its Numba type was made for.
_RECORDS = Members(RecordType, lambda record: record.cls, _field)


@cache
def record_type(cls: type[Struct]) -> type[RecordType]:
    """The device type of `cls`, made and given its struct model once."""
    kind = type(f"{cls.__name__}Type", (RecordType,), {"cls": cls})
    register_model(kind)(_Model)
    return kind


def register_member(
    cls: type[Struct], name: str, device: CUDADispatcher, *, attribute: bool
) -> None:
    """Make device function `device` the device member `name` of `cls`'s records.

    A dunder registers the operator it implements (`__getitem__` indexing, `__len__` `len`), an
    attribute is read without a call, and any other name is a method.
    """
    owner = record_type(cls)
    if name.startswith("__"):
        overload(_BUILTINS.get(name) or getattr(operator, name))(_overload(owner, device))
    else:
        _RECORDS.register(cls, name, device, attribute=attribute)


def _overload(owner: type[RecordType], device: CUDADispatcher) -> Callable:
    """The overload typing that answers, when its first argument is an `owner`, a function of
    `device`'s own parameters calling it.

    Numba's IR inliner takes no star arguments, so the parameters are spelled out, and the
    generated globals carry reserved names no parameter can shadow.
    """
    names = list(inspect.signature(device.py_func).parameters)
    spelled = ", ".join(names)
    namespace = {"_patos_owner": owner, "_patos_device": device}
    exec(
        f"def _patos_forwarded({spelled}):\n    return _patos_device({spelled})\n"
        f"def _patos_typing({spelled}):\n"
        f"    if isinstance({names[0]}, _patos_owner):\n        return _patos_forwarded\n",
        namespace,
    )
    return namespace["_patos_typing"]
