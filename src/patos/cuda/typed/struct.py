"""Records of device values, declared like a model, validated where the host builds them, and
passed to a kernel as one argument of their own device type."""

import annotationlib
from collections.abc import Callable, Collection, Mapping
from functools import cache, partial
from typing import TYPE_CHECKING, ClassVar, Self

import cupy as cp
import numpy as np

from .arguments import Argument, argument, record_argument
from .declarations import Declared, NamedValue, Record, declared, is_scalar, named
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
        if unknown := given.keys() - cls.declarations().keys():
            raise TypeError(f"{cls.__name__} has no field {', '.join(sorted(unknown))}")
        given = cls.__record_defaults__ | given
        problems = [f"{name} is missing" for name in fields if name not in given]
        held = self.__dict__
        for name, receive in cls._receivers():
            if name not in given:
                continue
            try:
                held[name] = receive(given[name])
            except (TypeError, ValueError, OverflowError) as error:
                problems.append(f"{name} {error}")
        if problems:
            raise TypeError(f"{cls.__name__}: {'; '.join(problems)}")
        object.__setattr__(self, "_argument", None)

    @classmethod
    @cache
    def _receivers(cls) -> tuple[tuple[str, Callable[[Value], Value]], ...]:
        """How each field, in order, takes a value: converted, checked or passed as it is."""
        return tuple((name, partial(_received, kind)) for name, kind in cls.declarations().items())

    @classmethod
    @cache
    def declarations(cls) -> dict[str, Declared]:
        """What each field declares, read once per record class."""
        annotations = annotationlib.get_annotations(cls)
        fields = {name: declared(annotations[name]) for name in cls.__record_fields__}
        if held := [name for name, kind in fields.items() if isinstance(kind, NamedValue)]:
            raise TypeError(f"{cls.__name__}.{', '.join(held)} is a named value, not a field")
        return fields

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
        roles = cls._roles()
        sized = {
            name: int(value)
            for name, value in values.items()
            if name in roles and isinstance(value, int | np.integer)
        }
        if stray := set(zeroed) - sized.keys():
            raise TypeError(f"{cls.__name__} zeroes {', '.join(sorted(stray))}, given no size")
        for name, size in sized.items():
            if (held := roles[name]) is None:
                raise TypeError(
                    f"{cls.__name__}.{name} declares {named(cls.declarations()[name])}, which has "
                    "no one dtype to take"
                )
            role, dtype = held
            taken = workspace.zeros if name in zeroed else workspace.take
            values[name] = taken(role, size, dtype)
        return cls(**values)

    @classmethod
    @cache
    def _roles(cls) -> dict[str, tuple[str, np.dtype] | None]:
        """The workspace role and dtype of each array field, None for an open element."""
        prefix = f"{cls.__module__}.{cls.__qualname__}"
        return {
            name: (f"{prefix}.{name}", np.dtype(kind.element)) if kind.concrete else None
            for name, kind in cls.declarations().items()
            if isinstance(kind, ArrayOf)
        }

    def __kernel_argument__(self) -> Argument:
        """What a kernel receives for this record, marshalled on its first launch."""
        held = self._argument
        if held is None:
            fields, constants = {}, []
            for name, kind in self.declarations().items():
                if isinstance(kind, ConstantOf):
                    constants.append((name, getattr(self, name)))
                    continue
                fields[name] = argument(getattr(self, name))
                if isinstance(kind, ArrayOf) and fields[name][0].layout != "C":
                    raise TypeError(f"{type(self).__name__}.{name} is strided, not contiguous")
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
    """Raise `TypeError` unless `value` is an array of what `kind` declares.

    Contiguity is checked where the record marshals, which reads every array's layout anyway.
    """
    dtype, shape = getattr(value, "dtype", None), getattr(value, "shape", None)
    if dtype is None or shape is None or not kind.admits(dtype, len(shape)):
        held = type(value).__name__ if dtype is None or shape is None else f"{dtype}{list(shape)}"
        raise TypeError(f"is {held}, not the {named(kind)} declared")


def _shown(value: Value) -> str:
    """An array by its element and shape, anything else by its repr."""
    shape, dtype = getattr(value, "shape", ()), getattr(value, "dtype", None)
    return f"{named(dtype.type)}{list(shape)}" if shape and dtype is not None else repr(value)
