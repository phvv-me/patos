"""Records of device values, declared like a model and validated where the host builds them.

`Record` is the declaration, read and checked without Numba. `patos.cuda.backend.struct.Struct`
extends it with what the device does with one: a device type of its own, its device members, the
argument a launch passes, and a constructor device code calls.
"""

import annotationlib
from collections.abc import Callable, Collection, Mapping
from functools import cache, partial
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np

from ..scalars import ArrayOf, ConstantOf, Lanes, converted
from .declarations import Declared, NamedValue, RecordOf, declared, is_scalar, named

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ..runtime.memory import Workspace
    from ..runtime.streams import StreamLike
    from ..scalars import Shaped

# What a record holds: a scalar, a host or device array, or another record.
type Value = bool | int | float | np.generic | np.ndarray | Shaped | Record


class Record:
    """A frozen record of device values, declared like a model.

    A subclass annotates its fields (`slots: Vector[u64]`, `mask: u64`, a default making a field
    optional). Building one validates every field: a scalar converts to its declared type, and an
    array of another element or dimension count, or a record of another class, fails, all of them
    reported at once. A record with fields takes no subclass.
    """

    __slots__ = ("__dict__",)
    __record_fields__: ClassVar[tuple[str, ...]] = ()
    __record_defaults__: ClassVar[dict[str, Value]] = {}

    @classmethod
    @cache
    def _dtypes(cls) -> dict[str, np.dtype | None]:
        """The dtype of each array field, None for an open element."""
        return {
            name: np.dtype(kind.element) if kind.concrete else None
            for name, kind in cls.declarations().items()
            if isinstance(kind, ArrayOf)
        }

    @classmethod
    @cache
    def _receivers(cls) -> tuple[tuple[str, Callable[[Value], Value]], ...]:
        """How each field, in order, takes a value: converted, checked or passed as it is."""
        return tuple(
            (name, partial(cls._received, kind)) for name, kind in cls.declarations().items()
        )

    @classmethod
    @cache
    def declarations(cls) -> dict[str, Declared]:
        """What each field declares, read once per record class."""
        annotations = annotationlib.get_annotations(cls)
        fields = {name: declared(annotations[name]) for name in cls.__record_fields__}
        if held := [name for name, kind in fields.items() if isinstance(kind, NamedValue | Lanes)]:
            raise TypeError(
                f"{cls.__name__}.{', '.join(held)} is a named value or lanes, which no launch "
                "passes, not a field"
            )
        return fields

    @classmethod
    def _given(cls, values: tuple[Value, ...], named: Mapping[str, Value]) -> dict[str, Value]:
        """The fields `values` and `named` give, by name; too many, repeated or unknown fail."""
        fields = cls.__record_fields__
        if len(values) > len(fields):
            raise TypeError(f"{cls.__name__} has {len(fields)} fields, {len(values)} given")
        given = dict(zip(fields, values, strict=False))
        if repeated := given.keys() & named.keys():
            raise TypeError(f"{cls.__name__} given {', '.join(repeated)} twice")
        given |= named
        if unknown := given.keys() - cls.declarations().keys():
            raise TypeError(f"{cls.__name__} has no field {', '.join(sorted(unknown))}")
        return given

    @classmethod
    def _received(cls, kind: Declared, value: Value) -> Value:
        """`value` as a field declaring `kind` holds it, or a `TypeError` saying why it cannot be.

        An integer converts with an overflow check and a float never truncates into an integer;
        an array is checked, not copied.
        """
        if isinstance(kind, ConstantOf):
            if not isinstance(value, int | np.integer):
                raise TypeError(
                    f"is a {type(value).__name__}, not the {kind.kind.__name__} declared"
                )
            return kind.kind(value)
        if kind is bool or is_scalar(kind):
            if not isinstance(value, int | float | np.number):
                raise TypeError(f"is a {type(value).__name__}, not the {named(kind)} declared")
            return converted(kind, value)
        if isinstance(kind, ArrayOf):
            cls._checked_array(kind, value)
        elif isinstance(kind, RecordOf) and not isinstance(value, kind.cls):
            raise TypeError(f"is a {type(value).__name__}, not the {kind.cls.__name__} declared")
        return value

    @classmethod
    def of(cls, source: Mapping[str, Value] | Record | BaseModel, **changes: Value) -> Self:
        """A record of the fields `source` holds by name, with `changes` applied.

        `source` is a mapping or holds them as attributes; a field it lacks takes its default.
        """
        names = cls.__record_fields__
        if isinstance(source, Mapping):
            held = {name: source[name] for name in names if name in source}
        else:
            held = {name: getattr(source, name) for name in names if hasattr(source, name)}
        return cls(**held | changes)

    @classmethod
    def take(
        cls,
        workspace: Workspace,
        *,
        stream: StreamLike,
        role: str | None = None,
        zeroed: Collection[str] = (),
        **values: Value,
    ) -> Self:
        """A record whose array fields given as a size are taken from `workspace`.

        Each such field is the buffer of the role `<role>.<field>`, in the dtype it declares,
        zero-filled on `stream` when `zeroed` names it; every other field is given as it is.

        role: the record's qualified name unless given. Two records of one class alive together
            must be taken under two roles, or they share their buffers.
        """
        dtypes = cls._dtypes()
        sized = {
            name: int(value)
            for name, value in values.items()
            if name in dtypes and isinstance(value, int | np.integer)
        }
        if stray := set(zeroed) - sized.keys():
            raise TypeError(f"{cls.__name__} zeroes {', '.join(sorted(stray))}, given no size")
        prefix = role or f"{cls.__module__}.{cls.__qualname__}"
        for name, size in sized.items():
            if (dtype := dtypes[name]) is None:
                raise TypeError(
                    f"{cls.__name__}.{name} declares {named(cls.declarations()[name])}, which has "
                    "no one dtype to take"
                )
            held = f"{prefix}.{name}"
            values[name] = (
                workspace.zeros(held, size, dtype, stream=stream)
                if name in zeroed
                else workspace.take(held, size, dtype)
            )
        return cls(**values)

    @staticmethod
    def _checked_array(kind: ArrayOf, value: Value) -> None:
        """Raise `TypeError` unless `value` is an array of what `kind` declares.

        Contiguity is checked where the record marshals, which reads every array's layout anyway.
        """
        dtype, shape = getattr(value, "dtype", None), getattr(value, "shape", None)
        if dtype is None or shape is None or not kind.admits(dtype, len(shape)):
            held = f"{dtype}{list(shape)}" if dtype is not None and shape is not None else None
            raise TypeError(f"is {held or type(value).__name__}, not the {named(kind)} declared")

    @staticmethod
    def _shown(value: Value) -> str:
        """An array by its element and shape, anything else by its repr."""
        shape, dtype = getattr(value, "shape", ()), getattr(value, "dtype", None)
        return f"{named(dtype.type)}{list(shape)}" if shape and dtype is not None else repr(value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is frozen")

    def __init__(self, *values: Value, **named: Value) -> None:
        cls = type(self)
        given = cls.__record_defaults__ | cls._given(values, named)
        problems = [f"{name} is missing" for name in cls.__record_fields__ if name not in given]
        for name, receive in cls._receivers():
            if name not in given:
                continue
            try:
                self.__dict__[name] = receive(given[name])
            except (TypeError, ValueError, OverflowError) as error:
                problems.append(f"{name} {error}")
        if problems:
            raise TypeError(f"{cls.__name__}: {'; '.join(problems)}")

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__record_fields__:
            raise TypeError(f"{cls.__name__} extends a record, which takes no subclass")
        own = annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF)
        namespace = vars(cls)
        cls.__record_fields__ = tuple(own)
        cls.__record_defaults__ = {name: namespace[name] for name in own if name in namespace}
        cls.__match_args__ = cls.__record_fields__

    def __reduce__(self) -> tuple:
        return type(self).of, (vars(self),)

    def __replace__(self, **changes: Value) -> Self:
        """A copy with `changes` applied, validated as a new record is (`copy.replace`)."""
        return type(self)(**vars(self) | changes)

    def __repr__(self) -> str:
        shown = ", ".join(f"{name}={self._shown(value)}" for name, value in vars(self).items())
        return f"{type(self).__name__}({shown})"

    def __setattr__(self, name: str, value: Value) -> None:
        raise AttributeError(f"{type(self).__name__} is frozen; copy.replace it")
