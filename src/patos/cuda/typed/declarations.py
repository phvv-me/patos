"""What an annotation declares, read through `type` aliases at every level.

A declaration is a scalar type, `bool`, an `Array[T]`, a record (a named tuple whose fields are
annotated, such as a `Struct`), or a tuple of these. The readings of a function also know an int
literal, which takes the type of the integer it meets.
"""

import annotationlib
from collections.abc import Callable
from dataclasses import dataclass
from types import GenericAlias, ModuleType
from typing import TypeAliasType, TypeIs, TypeVar, final, get_args, get_origin

import numpy as np
from numba import types

from .scalars import Array, i16, i32, i64, u8, u16, u32, u64


@final
@dataclass(frozen=True)
class Literal:
    """An int constant written in the function body, taking the type of the integer it meets."""

    value: int


@final
@dataclass(frozen=True)
class ArrayOf:
    """An `Array[T]` annotation, its element None for a type parameter or an abstract type."""

    element: type[np.generic] | None


@final
@dataclass(frozen=True)
class Record:
    """A named-tuple annotation: the record type and what each of its fields declares."""

    cls: type
    fields: tuple[tuple[str, Declared], ...]

    def field(self, name: str) -> Declared:
        return dict(self.fields).get(name)


# What a name in a function's globals or closure, or an annotation, evaluates to; a name not yet
# defined stays a forward reference.
type Evaluated = (
    type
    | TypeAliasType
    | TypeVar
    | GenericAlias
    | annotationlib.ForwardRef
    | ModuleType
    | Callable
    | int
    | np.generic
    | None
)
# What a reading knows of a value: its scalar type, `bool`, a literal, or None for unknown.
type Kind = type[np.generic] | type[bool] | Literal | None
# What an annotation declares: a kind, an array, a record, or a tuple of these.
type Declared = Kind | ArrayOf | Record | tuple[Declared, ...]
type Returns = Declared | type[None]
# Anything the predicates below are asked about.
type Subject = Evaluated | Returns


@final
@dataclass(frozen=True)
class Signature:
    """What a compiled device function declares, for its callers' checks."""

    parameters: tuple[Declared, ...]
    returns: Returns


_SHORT: dict[Returns, str] = {
    i16: "i16", i32: "i32", i64: "i64", u8: "u8", u16: "u16", u32: "u32", u64: "u64",
}  # fmt: skip
_CONCRETE = frozenset(np.sctypeDict.values())


def is_scalar(kind: Subject) -> TypeIs[type[np.number]]:
    return isinstance(kind, type) and issubclass(kind, np.number) and kind in _CONCRETE


def is_integer(kind: Subject) -> TypeIs[type[np.integer]]:
    return is_scalar(kind) and issubclass(kind, np.integer)


def named(kind: Returns) -> str:
    if isinstance(kind, Record):
        return kind.cls.__name__
    if isinstance(kind, tuple):
        return f"tuple[{', '.join(named(element) for element in kind)}]"
    return _SHORT.get(kind) or getattr(kind, "__name__", str(kind))


def unaliased(value: Evaluated) -> Evaluated:
    while isinstance(value, TypeAliasType):
        value = value.__value__
    return value


def declared(value: Evaluated) -> Declared:
    """Return what an annotation already evaluated declares, through type aliases at every level.

    None names no device type.
    """
    value = unaliased(value)
    if value is bool:
        return bool
    if isinstance(value, type) and issubclass(value, tuple) and hasattr(value, "_fields"):
        annotations = annotationlib.get_annotations(value, format=annotationlib.Format.FORWARDREF)
        return Record(value, tuple((name, declared(kind)) for name, kind in annotations.items()))
    if is_scalar(value):
        return value
    return _declared_generic(get_origin(value), get_args(value))


def numba_type(declared: Returns) -> types.Type | None:
    """The Numba type a scalar, `bool` or a tuple of these declares; None for anything else."""
    if isinstance(declared, tuple):
        elements = [numba_type(element) for element in declared]
        return None if None in elements else types.BaseTuple.from_types(elements)
    if is_scalar(declared):
        return getattr(types, np.dtype(declared).name)
    return types.boolean if declared is bool else None


def _declared_generic(origin: Evaluated, arguments: tuple[Evaluated, ...]) -> Declared:
    """Return what a subscripted annotation declares: an alias, a tuple or an array.

    An array of a type parameter or of an abstract NumPy type declares no element.
    """
    if isinstance(origin, TypeAliasType):
        return declared(origin.__value__[arguments])
    if origin is tuple:
        return tuple(declared(element) for element in arguments)
    if origin is Array:
        return _array_of(unaliased(arguments[0]))
    return None


def _array_of(element: Evaluated) -> ArrayOf | None:
    """An array of `element`, which a type parameter or an abstract NumPy type leaves open."""
    if is_scalar(element):
        return ArrayOf(element)
    if isinstance(element, TypeVar) or (
        isinstance(element, type) and issubclass(element, np.generic)
    ):
        return ArrayOf(None)
    return None
