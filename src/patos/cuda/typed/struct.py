"""Records of device values, declared like a model and passed to a kernel as one argument."""

import annotationlib
import collections
from typing import TYPE_CHECKING, Self

import numpy as np

from .declarations import ArrayOf, Evaluated, declared, is_scalar, named

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class _StructMeta(type):
    """Make every subclass of `Struct` a named tuple of its annotated fields."""

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict, **kwargs) -> type:
        draft = super().__new__(mcs, name, bases, namespace, **kwargs)
        if not any(isinstance(base, _StructMeta) for base in bases):
            return draft
        fields = annotationlib.get_annotations(draft, format=annotationlib.Format.FORWARDREF)
        defaults = [namespace[field] for field in fields if field in namespace]
        record = collections.namedtuple(name, list(fields), defaults=defaults or None)
        # A field's default lives in the record, and the draft's own `__dict__` would shadow the
        # tuple's slots.
        members = {
            key: value for key, value in namespace.items() if key not in {*fields, "__dict__"}
        }

        def __new__(cls: type[Struct], *arguments: ArrayLike, **keywords: ArrayLike) -> Struct:
            values = record(*arguments, **keywords)
            return tuple.__new__(cls, map(cls._received, fields, values, strict=True))

        members |= {"__slots__": (), "__new__": __new__, "_declarations": fields}
        # `Struct` ahead of the record, so its checked `_replace` wins over the record's own, which
        # rebuilds through `tuple.__new__` past the checks.
        return super().__new__(mcs, name, (*bases, record), members, **kwargs)


class Struct(tuple, metaclass=_StructMeta):
    """A frozen record of device values, declared like a model and passed to a kernel as one
    argument.

    A subclass annotates its fields (`slots: Array[Word]`, `mask: Word`, a default making a field
    optional) and becomes a named tuple, which numba passes by value and reads by attribute in
    device code. Building one checks every array's dtype and converts every scalar to its declared
    type, so a wrong array fails where the host built the record rather than inside numba's
    typing, and a kernel reads `tables.slots` with every type already exact.
    """

    __slots__ = ()
    _declarations: dict[str, Evaluated]

    if TYPE_CHECKING:

        def __new__(cls, *arguments: ArrayLike, **keywords: ArrayLike) -> Self: ...

        _fields: tuple[str, ...]

        def _asdict(self) -> dict[str, ArrayLike]: ...

    @classmethod
    def _received(cls, field: str, value: ArrayLike) -> ArrayLike:
        """`value` as field `field` declares it: a scalar converted, an array's dtype checked."""
        kind = declared(cls._declarations[field])
        if kind is bool:
            return bool(value)
        if is_scalar(kind):
            return np.asarray(value, dtype=kind)[()]
        if isinstance(kind, ArrayOf) and kind.element is not None:
            dtype = np.dtype(getattr(value, "dtype", None))
            if dtype != np.dtype(kind.element):
                expected = named(kind.element)
                raise TypeError(
                    f"{cls.__name__}.{field} holds {dtype}, not the {expected} declared"
                )
        return value

    def _replace(self, **changes: ArrayLike) -> Self:
        """A copy with `changes` applied, checked as a new record is."""
        return type(self)(**{**self._asdict(), **changes})
