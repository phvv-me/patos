"""A record on the device: its own Numba type, a C struct of its fields, the device methods,
operators and properties its class declares, and its class called in device code to build one.

A record is no tuple to Numba, so `len`, indexing and `in` mean only what its class declares.
"""

import inspect
import operator
from collections.abc import Callable, Sequence
from functools import cache
from typing import TYPE_CHECKING, ClassVar

from llvmlite import ir
from numba import types
from numba.core.errors import TypingError
from numba.cuda import cgutils
from numba.cuda.core import imputils
from numba.cuda.cudadecl import registry as typing_registry
from numba.cuda.cudaimpl import registry as lowering_registry
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda.extending import core_models, models, overload, register_model
from numba.cuda.typing import templates

from ..front.declarations import Declared, RecordOf, named
from ..scalars import ConstantOf
from .declared import is_admitted, numba_type
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


class RecordClass(types.Opaque):
    """The device type of a record class, which device code calls to build a record."""

    @property
    def key(self) -> type[Struct]:
        return self.cls

    def __init__(self, cls: type[Struct]) -> None:
        self.cls = cls
        super().__init__(f"class({cls.__module__}.{cls.__qualname__})")


register_model(RecordClass)(core_models.OpaqueModel)


@typing_registry.register_attr
class _Constructor(templates.AttributeTemplate):
    key = RecordClass

    def resolve___call__(self, classty: RecordClass) -> types.Function:
        """The call building a record: every field given, a scalar converting to its declaration.

        patos's rewrite fills in a field's default and passes every field by position.
        """
        cls = classty.cls
        fields = inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in cls.__record_fields__
            ]
        )

        def typer(*given: types.Type, **named: types.Type) -> RecordType:
            try:
                bound = fields.bind(*given, **named)
            except TypeError as error:
                raise TypingError(f"{cls.__name__}: {error}") from None
            return _built(cls, bound.args)

        typer.__dict__["pysig"] = fields
        return types.Function(templates.make_callable_template(self.key, typer))


@typing_registry.register_attr
class _Fields(templates.AttributeTemplate):
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
    return imputils.impl_ret_borrowed(
        context, builder, record.members[attr], getattr(fields, attr)
    )


# A record's class is the one its Numba type was made for.
_RECORDS = Members(RecordType, lambda record: record.cls, _field)


@cache
def record_type(cls: type[Struct]) -> type[RecordType]:
    """The device type of `cls`, made and given its struct model once."""
    kind = type(f"{cls.__name__}Type", (RecordType,), {"cls": cls})
    register_model(kind)(_Model)
    return kind


def _built(cls: type[Struct], given: Sequence[types.Type]) -> RecordType:
    """The type of a `cls` record device code builds from fields of the types `given`.

    A scalar field holds its declared type, an array or a record the one given, and a constant,
    part of the type, is given as an int literal.
    """
    fields, constants = [], []
    for (name, kind), held in zip(cls.declarations().items(), given, strict=True):
        if not isinstance(kind, ConstantOf):
            fields.append((name, _field_type(cls, name, kind, held)))
        elif isinstance(held, types.IntegerLiteral):
            constants.append((name, kind.kind(held.literal_value)))
        else:
            raise TypingError(
                f"{cls.__name__}.{name} is a constant, part of the record's type, so it takes an "
                f"int literal, not {held}"
            )
    return record_type(cls)(tuple(fields), tuple(constants))


def _field_type(cls: type[Struct], name: str, kind: Declared, given: types.Type) -> types.Type:
    """The type field `name` holds when it is built from a `given`."""
    if (declared := numba_type(kind)) is not None:
        return declared
    held = types.unliteral(given)
    if isinstance(kind, RecordOf):
        fits = isinstance(held, RecordType) and held.cls is kind.cls
    else:
        fits = is_admitted(kind, held)
    if not fits:
        raise TypingError(
            f"{cls.__name__}.{name} receives {given} where {named(kind)} is declared"
        )
    return held


@lowering_registry.lower(RecordClass, types.VarArg(types.Any))
def _constructed(context, builder: ir.IRBuilder, call: templates.Signature, values: Sequence):
    """Build the record `call` types, each field cast to the type it holds."""
    record = call.return_type
    fields = cgutils.create_struct_proxy(record)(context, builder)
    for name, given, value in zip(record.cls.__record_fields__, call.args, values, strict=True):
        if name in record.members:
            setattr(fields, name, context.cast(builder, value, given, record.members[name]))
    return imputils.impl_ret_borrowed(context, builder, record, fields._getvalue())


def register_constructor(cls: type[Struct]) -> None:
    """Let device code call `cls` to build a record."""
    typing_registry.register_global(cls, RecordClass(cls))


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
    """The overload typing that answers a call whose first argument is an `owner`.

    It answers with a function of `device`'s own parameters calling it.

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
