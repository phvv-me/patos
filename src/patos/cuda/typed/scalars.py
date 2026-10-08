"""The numeric types device code spells. Each is a scalar type and, subscripted by its shape, an
array of it: `u8[int]` a vector, `i16[int, int]` a table, as numba spells `uint8[:]` and
`int16[:, :]`. Called, a type converts a value to its scalar (`u64(first)`).

`number` and `unsigned` are open element types, for an array whose element is any number or any
unsigned integer (`unsigned[int]`).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self, SupportsInt, cast, final, overload

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from ..runtime.streams import Stream


class Shaped(Protocol):
    """An array of any module, host or device, known by its shape and its dtype."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...


@final
@dataclass(frozen=True)
class ArrayOf:
    """An array annotation: its element type, concrete or open, and its dimension count.

    An array is contiguous, so every kernel compiles one signature.
    """

    element: type[np.number]
    ndim: int

    @property
    def concrete(self) -> bool:
        """Whether the element is one scalar type, which fixes the array's dtype."""
        return self.element in _CONCRETE

    def admits(self, dtype: np.dtype, ndim: int) -> bool:
        """Whether an array of `dtype` and `ndim` dimensions is one this annotation declares."""
        if ndim != self.ndim:
            return False
        return dtype == self.element if self.concrete else np.issubdtype(dtype, self.element)


@final
@dataclass(frozen=True)
class ConstantOf:
    """A record field whose value is part of the record's device type, compiled as a literal."""

    kind: type[int] | type[bool]


if TYPE_CHECKING:
    type Constant[T] = T
else:

    class Constant:
        """A record field whose value is part of the record's device type: `width: Constant[int]`.

        Device code reads it as a literal, so it sizes a local array or unrolls a loop the way a
        closure constant does, and each value compiles its own kernels.
        """

        def __class_getitem__(cls, kind):
            return ConstantOf(kind)


def converted(
    kind: type[np.number] | type[bool], value: int | float | np.number
) -> np.number | bool:
    """`value` as scalar `kind`, an integer checked for overflow and a float never truncated into
    an integer; `TypeError` or `OverflowError` says why it cannot be."""
    if kind is bool:
        return bool(value)
    if isinstance(value, np.number) and type(value) is kind:
        return value
    spelled = SPELLINGS.get(kind, kind.__name__)
    if isinstance(value, int | np.integer):
        try:
            return kind(int(value))
        except OverflowError:
            raise OverflowError(f"is {value}, out of range of the {spelled} declared") from None
    if isinstance(value, float | np.floating) and not issubclass(kind, np.integer):
        return cast("np.number", np.asarray(value, dtype=kind)[()])
    raise TypeError(f"is a {type(value).__name__}, not the {spelled} declared")


_CONCRETE = frozenset(np.sctypeDict.values())
# How patos spells the numpy types it names, in messages as in annotations.
SPELLINGS: dict[type, str] = {
    np.int16: "i16", np.int32: "i32", np.int64: "i64", np.uint8: "u8", np.uint16: "u16",
    np.uint32: "u32", np.uint64: "u64", np.unsignedinteger: "unsigned", np.number: "number",
}  # fmt: skip

if TYPE_CHECKING:

    class Numeric[T, *Shape]:
        """What a type checker knows of a numeric value or of an array of them.

        Device code computes with the scalars; host code reads a record's arrays as the numpy or
        cupy arrays they hold, so an array answers what both modules' arrays do. An element read
        out of an array is its patos scalar `T`.
        """

        size: int
        shape: tuple[int, ...]
        ndim: int
        dtype: np.dtype
        nbytes: int

        @overload
        def __getitem__(self, index: Subscript | tuple[Subscript, ...]) -> T: ...

        @overload
        def __getitem__(self, index: slice | Shaped) -> Self: ...

        def __getitem__(
            self, index: Subscript | tuple[Subscript, ...] | slice | Shaped
        ) -> T | Self: ...

        def __setitem__(
            self, index: Subscript | tuple[Subscript, ...] | slice | Shaped, value: Operand
        ) -> None: ...

        def __len__(self) -> int: ...

        def __add__(self, value: Operand) -> Self: ...
        def __radd__(self, value: Operand) -> Self: ...
        def __sub__(self, value: Operand) -> Self: ...
        def __rsub__(self, value: Operand) -> Self: ...
        def __mul__(self, value: Operand) -> Self: ...
        def __rmul__(self, value: Operand) -> Self: ...
        def __floordiv__(self, value: Operand) -> Self: ...
        def __mod__(self, value: Operand) -> Self: ...
        def __and__(self, value: Operand) -> Self: ...
        def __rand__(self, value: Operand) -> Self: ...
        def __or__(self, value: Operand) -> Self: ...
        def __ror__(self, value: Operand) -> Self: ...
        def __xor__(self, value: Operand) -> Self: ...
        def __lshift__(self, value: Operand) -> Self: ...
        def __rshift__(self, value: Operand) -> Self: ...
        def __rxor__(self, value: Operand) -> Self: ...
        def __rlshift__(self, value: Operand) -> Self: ...
        def __rrshift__(self, value: Operand) -> Self: ...
        def __rfloordiv__(self, value: Operand) -> Self: ...
        def __rmod__(self, value: Operand) -> Self: ...
        def __neg__(self) -> Self: ...
        def __invert__(self) -> Self: ...
        def __lt__(self, value: Operand) -> bool: ...
        def __le__(self, value: Operand) -> bool: ...
        def __gt__(self, value: Operand) -> bool: ...
        def __ge__(self, value: Operand) -> bool: ...
        def __int__(self) -> int: ...
        def __index__(self) -> int: ...

        def astype(self, dtype: DTypeLike) -> number[*Shape]: ...

        def view(self, dtype: DTypeLike) -> number[*Shape]: ...

        def copy(self) -> Self: ...

        def fill(self, value: int) -> None: ...

        def get(self, *, stream: Stream | None = None) -> np.ndarray: ...

    # What arithmetic meets a numeric value with, and what indexes an array.
    type Operand = int | np.integer | Shaped | Numeric
    type Subscript = int | np.integer | u8 | u16 | u32 | u64 | i16 | i32 | i64

    # A type called converts to its scalar, which an array's elements are as well.
    class u8[*Shape = *tuple[()]](Numeric["u8", *Shape]):
        def __new__(cls, value: SupportsInt = 0) -> u8: ...

    class u16[*Shape = *tuple[()]](Numeric["u16", *Shape]):
        def __new__(cls, value: SupportsInt = 0) -> u16: ...

    class u32[*Shape = *tuple[()]](Numeric["u32", *Shape]):
        def __new__(cls, value: SupportsInt = 0) -> u32: ...

    class u64[*Shape = *tuple[()]](Numeric["u64", *Shape]):
        def __new__(cls, value: SupportsInt = 0) -> u64: ...

    class i16[*Shape = *tuple[()]](Numeric["i16", *Shape]):
        def __new__(cls, value: SupportsInt = 0) -> i16: ...

    class i32[*Shape = *tuple[()]](Numeric["i32", *Shape]):
        def __new__(cls, value: SupportsInt = 0) -> i32: ...

    class i64[*Shape = *tuple[()]](Numeric["i64", *Shape]):
        def __new__(cls, value: SupportsInt = 0) -> i64: ...

    class unsigned[*Shape](Numeric["unsigned", *Shape]): ...

    class number[*Shape](Numeric["number", *Shape]): ...

else:

    def _numeric(name: str, base: type[np.number]) -> type:
        """`base` as a numeric type: called, its scalar; subscripted by a shape, an array of it.

        A subclass, so numba types a cast through it as one of `base`.
        """

        def __new__(cls, value=0):
            return base(value)

        def __class_getitem__(cls, shape):
            dimensions = shape if isinstance(shape, tuple) else (shape,)
            if not dimensions or any(dimension is not int for dimension in dimensions):
                raise TypeError(f"{name}[{shape!r}]: an array names each dimension `int`")
            return ArrayOf(base, len(dimensions))

        members = {"__slots__": (), "__new__": __new__, "__module__": __name__}
        return type(name, (base,), members | {"__class_getitem__": classmethod(__class_getitem__)})

    u8, u16, u32, u64 = (_numeric(f"u{n}", getattr(np, f"uint{n}")) for n in (8, 16, 32, 64))
    i16, i32, i64 = (_numeric(f"i{n}", getattr(np, f"int{n}")) for n in (16, 32, 64))
    unsigned = _numeric("unsigned", np.unsignedinteger)
    number = _numeric("number", np.number)


def canonical(kind: type[np.number]) -> type[np.number]:
    """The numpy type a numeric type names, which is what patos reads and numba types."""
    return next(
        base for base in kind.__mro__ if issubclass(base, np.number) and base.__module__ == "numpy"
    )
