"""What an annotation declares, read through `type` aliases at every level.

A declaration is a scalar type, `bool`, an array (`u8[int]`), a record (a `Struct`), or a tuple
of these. The readings of a function also know an int literal, which takes the type of the
integer it meets.
"""

import annotationlib
from collections.abc import Callable
from dataclasses import dataclass
from types import GenericAlias, ModuleType
from typing import Protocol, TypeAliasType, TypeIs, TypeVar, final, get_args, get_origin

import numpy as np
from numba import types

from .scalars import SPELLINGS, ArrayOf, ConstantOf, canonical


@final
@dataclass(frozen=True)
class IntLiteral:
    """An int constant written in the function body, taking the type of the integer it meets."""

    value: int


@final
@dataclass(frozen=True)
class Record:
    """A `Struct` annotation."""

    cls: type[_Recorded]

    def field(self, name: str) -> Declared:
        """What field `name` declares, None for a name the record lacks."""
        return self.cls.declarations().get(name)


# What a name in a function's globals or closure, or an annotation, evaluates to; a name not yet
# defined stays a forward reference.
type Evaluated = (
    type
    | TypeAliasType
    | TypeVar
    | GenericAlias
    | ArrayOf
    | ConstantOf
    | annotationlib.ForwardRef
    | ModuleType
    | Callable
    | int
    | np.generic
    | None
)
# What a reading knows of a value: its scalar type, `bool`, a literal, or None for unknown.
type Kind = type[np.generic] | type[bool] | IntLiteral | None
# What an annotation declares: a kind, an array, a record, or a tuple of these.
type Declared = Kind | ArrayOf | ConstantOf | Record | tuple[Declared, ...]
type Returns = Declared | type[None]
# Anything the predicates below are asked about.
type Subject = Evaluated | Returns


@final
@dataclass(frozen=True)
class Signature:
    """What a compiled device function declares, for its callers' checks."""

    parameters: tuple[Declared, ...]
    returns: Returns


class _Recorded(Protocol):
    __record_fields__: tuple[str, ...]

    @classmethod
    def declarations(cls) -> dict[str, Declared]: ...


def is_scalar(kind: Subject) -> TypeIs[type[np.number]]:
    """Whether `kind` is one numeric type, as numpy names it or as patos does."""
    return isinstance(kind, type) and issubclass(kind, np.number) and canonical(kind) in _SCALARS


def is_integer(kind: Subject) -> TypeIs[type[np.integer]]:
    return is_scalar(kind) and issubclass(kind, np.integer)


def named(kind: Returns) -> str:
    match kind:
        case Record():
            return kind.cls.__name__
        case ArrayOf():
            return f"{named(kind.element)}[{', '.join(['int'] * kind.ndim)}]"
        case tuple():
            return f"tuple[{', '.join(named(element) for element in kind)}]"
        case type():
            return SPELLINGS.get(kind, kind.__name__)
    return str(kind)


def unaliased(value: Evaluated) -> Evaluated:
    while isinstance(value, TypeAliasType):
        value = value.__value__
    return value


def declared(value: Evaluated) -> Declared:
    """Return what an annotation already evaluated declares, through type aliases at every level.

    A scalar declares as the numpy type it names. None names no device type.
    """
    value = unaliased(value)
    if value is bool or isinstance(value, ArrayOf | ConstantOf):
        return value
    if _is_record(value):
        return Record(value)
    if is_scalar(value):
        return canonical(value)
    origin = get_origin(value)
    if isinstance(origin, TypeAliasType):
        return declared(origin.__value__[get_args(value)])
    if origin is tuple:
        return tuple(declared(element) for element in get_args(value))
    return None


def numba_type(declared: Returns) -> types.Type | None:
    """The Numba type a scalar, `bool` or a tuple of these declares; None for anything else."""
    if isinstance(declared, tuple):
        elements = [numba_type(element) for element in declared]
        return None if None in elements else types.BaseTuple.from_types(elements)
    if is_scalar(declared):
        return getattr(types, np.dtype(declared).name)
    return types.boolean if declared is bool else None


_SCALARS = frozenset(
    {np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64}
    | {np.float16, np.float32, np.float64}
)


def _is_record(value: Subject) -> TypeIs[type[_Recorded]]:
    return isinstance(value, type) and hasattr(value, "__record_fields__")
