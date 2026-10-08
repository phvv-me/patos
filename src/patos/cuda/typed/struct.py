"""Records of device values, declared like a model, validated where the host builds them, and
passed to a kernel as one argument of their own device type."""

import annotationlib
from collections.abc import Collection, Mapping
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Self

import cupy as cp
import numpy as np

from .arguments import Argument, argument, record_argument
from .declarations import Declared, Record, declared, is_scalar, named
from .decorators import Method
from .kernels import Kernel
from .scalars import ArrayOf, ConstantOf, converted

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..runtime.memory import Workspace
    from .scalars import Shaped

# What a record holds: a scalar, a host or device array, or another record.
type Value = bool | int | float | np.generic | np.ndarray | Shaped | Struct


class Struct:
    """A frozen record of device values, declared like a model and passed to a kernel as one
    argument.

    A subclass annotates its fields (`slots: u64[int]`, `mask: u64`, a default making a field
    optional). Building one validates every field: a scalar converts to its declared type, a host
    array uploads, and an array of another element or dimension count, or a record of another
    class, fails, all of them reported at once. Numba sees a record as a C struct of its fields,
    read by attribute in device code. A record with fields takes no subclass.

    Device functions defined in the class with an unannotated `self` are its device members: a
    method, an operator when the name is a dunder (`__getitem__`, `__len__`, `__contains__`), or
    an attribute under `@property`; the host class keeps none of them. A `kernel` defined so is
    launched from a record, `record.kernel[items](*arguments)`.
    """

    __slots__ = ("__dict__", "_argument")
    _argument: Argument | None
    __record_fields__: ClassVar[tuple[str, ...]] = ()
    __record_defaults__: ClassVar[dict[str, Value]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__record_fields__:
            raise TypeError(f"{cls.__name__} extends a record, which takes no subclass")
        own = annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF)
        namespace = vars(cls)
        cls.__record_fields__ = tuple(own)
        cls.__record_defaults__ = {name: namespace[name] for name in own if name in namespace}
        cls.__match_args__ = cls.__record_fields__
        for name, member in list(namespace.items()):
            if isinstance(member, property) and isinstance(member.fget, Method):
                member.fget.bind(cls, name, attribute=True)
            elif isinstance(member, Method) or (isinstance(member, Kernel) and member.member):
                member.bind(cls, name)

    def __init__(self, *values: Value, **named: Value) -> None:
        cls = type(self)
        fields = cls.__record_fields__
        if len(values) > len(fields):
            raise TypeError(f"{cls.__name__} has {len(fields)} fields, {len(values)} given")
        given = dict(zip(fields, values, strict=False))
        if repeated := given.keys() & named.keys():
            raise TypeError(f"{cls.__name__} given {', '.join(repeated)} twice")
        given |= named
        if unknown := given.keys() - set(fields):
            raise TypeError(f"{cls.__name__} has no field {', '.join(sorted(unknown))}")
        given = cls.__record_defaults__ | given
        problems = [f"{name} is missing" for name in fields if name not in given]
        for name, kind in cls.declarations().items():
            if name not in given:
                continue
            try:
                given[name] = _received(kind, given[name])
            except (TypeError, ValueError, OverflowError) as error:
                problems.append(f"{name} {error}")
        if problems:
            raise TypeError(f"{cls.__name__}: {'; '.join(problems)}")
        self.__dict__.update((name, given[name]) for name in fields)
        object.__setattr__(self, "_argument", None)

    @classmethod
    @cache
    def declarations(cls) -> dict[str, Declared]:
        """What each field declares, read once per record class."""
        annotations = annotationlib.get_annotations(cls)
        return {name: declared(annotations[name]) for name in cls.__record_fields__}

    @classmethod
    def of(cls, source: Mapping[str, Value] | Struct | BaseModel, **changes: Value) -> Self:
        """A record of the fields `source` holds by name, as a mapping or as attributes, with
        `changes` applied; a field `source` lacks takes its default."""
        names = cls.__record_fields__
        if isinstance(source, Mapping):
            held = {name: source[name] for name in names if name in source}
        else:
            held = {name: getattr(source, name) for name in names if hasattr(source, name)}
        return cls(**held | changes)

    @classmethod
    def take(cls, workspace: Workspace, *, zeroed: Collection[str] = (), **values: Value) -> Self:
        """A record whose array fields given as a size are taken from `workspace`.

        Each such field is the buffer of its role, the record's qualified name and the field's,
        in the dtype it declares, zero-filled when `zeroed` names it; every other field is given
        as it is.
        """
        kinds = cls.declarations()
        sized = {
            name: int(value)
            for name, value in values.items()
            if isinstance(value, int | np.integer) and isinstance(kinds[name], ArrayOf)
        }
        if stray := set(zeroed) - sized.keys():
            raise TypeError(f"{cls.__name__} zeroes {', '.join(sorted(stray))}, given no size")
        role = f"{cls.__module__}.{cls.__qualname__}"
        for name, size in sized.items():
            taken = workspace.zeros if name in zeroed else workspace.take
            values[name] = taken(f"{role}.{name}", size, _element(kinds[name]))
        return cls(**values)

    def __kernel_argument__(self) -> Argument:
        """What a kernel receives for this record, marshalled on its first launch."""
        held = self._argument
        if held is None:
            fields, constants = {}, []
            for name, kind in self.declarations().items():
                if isinstance(kind, ConstantOf):
                    constants.append((name, getattr(self, name)))
                else:
                    fields[name] = argument(getattr(self, name))
            held = record_argument(type(self), fields, tuple(constants))
            object.__setattr__(self, "_argument", held)
        return held

    def __replace__(self, **changes: Value) -> Self:
        """A copy with `changes` applied, validated as a new record is (`copy.replace`)."""
        return type(self)(**vars(self) | changes)

    def __reduce__(self) -> tuple:
        return type(self).of, (vars(self),)

    def __repr__(self) -> str:
        shown = ", ".join(f"{name}={_shown(value)}" for name, value in vars(self).items())
        return f"{type(self).__name__}({shown})"

    def __setattr__(self, name: str, value: Value) -> None:
        raise AttributeError(f"{type(self).__name__} is frozen; copy.replace it")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is frozen")


def _received(kind: Declared, value: Value) -> Value:
    """`value` as a field declaring `kind` holds it, or a `TypeError` saying why it cannot be.

    An integer converts with an overflow check and a float never truncates into an integer; an
    array is checked before a host one uploads.
    """
    if isinstance(kind, ConstantOf):
        if not isinstance(value, int | np.integer):
            raise TypeError(f"is a {type(value).__name__}, not the {kind.kind.__name__} declared")
        return kind.kind(value)
    if kind is bool or is_scalar(kind):
        if not isinstance(value, int | float | np.number):
            raise TypeError(f"is a {type(value).__name__}, not the {named(kind)} declared")
        return converted(kind, value)
    if isinstance(kind, ArrayOf):
        _check_array(kind, value)
        return cp.asarray(value) if isinstance(value, np.ndarray) else value
    if isinstance(kind, Record) and not isinstance(value, kind.cls):
        raise TypeError(f"is a {type(value).__name__}, not the {kind.cls.__name__} declared")
    return value


def _check_array(kind: ArrayOf, value: Value) -> None:
    """Raise `TypeError` unless `value` is a contiguous array of what `kind` declares."""
    dtype, shape = getattr(value, "dtype", None), getattr(value, "shape", None)
    if dtype is None or shape is None or not kind.admits(dtype, len(shape)):
        held = type(value).__name__ if dtype is None or shape is None else f"{dtype}{list(shape)}"
        raise TypeError(f"is {held}, not the {named(kind)} declared")
    if isinstance(value, np.ndarray | cp.ndarray) and not value.flags.c_contiguous:
        raise TypeError(f"is strided, where {named(kind)} is a contiguous array")


def _element(kind: Declared) -> type[np.number]:
    """The dtype an array field is taken in, which only a concrete element names."""
    if isinstance(kind, ArrayOf) and kind.concrete:
        return kind.element
    raise TypeError(f"a {named(kind)} field has no one dtype to take")


def _shown(value: Value) -> str:
    """An array by its element and shape, anything else by its repr."""
    shape, dtype = getattr(value, "shape", ()), getattr(value, "dtype", None)
    return f"{named(dtype.type)}{list(shape)}" if shape and dtype is not None else repr(value)
