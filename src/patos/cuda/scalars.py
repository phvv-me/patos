"""The numeric types device code spells, importable where no CUDA stack is installed. Each is a
scalar type: called, it converts a value to its scalar (`u64(first)`). `Vector[T]` is an array of
one dimension of the numeric type `T` and `Matrix[T]` one of two, as numba spells `uint8[:]` and
`int16[:, :]`: `Vector[u8]`, `Matrix[i16]`.

`number` and `unsigned` are open element types, for an array whose element is any number or any
unsigned integer (`Vector[unsigned]`), and so is a type parameter (`Vector[T]`), which a checker
reads as the parameter, so that two arrays of one `T` hold one type. `u8x4`, `i8x4`, `u16x2` and
`i16x2` are packed lanes, a 32-bit register read as four bytes or two halves, which
`primitives.bits` picks its instruction by.

A type checker reads the scalars as Python's `int` and `float`, since patos converts integers into
one another as C does, which no checker models. So the checker cannot tell `Vector[u8]` from
`Vector[u32]`, or `u8` from `i32`; patos refuses the wrong one where a kernel compiles or launches.
"""

from typing import TYPE_CHECKING, Protocol, Self, SupportsInt, TypeVar, cast, final, overload

import numpy as np

from ..bases import FrozenModel

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from .runtime.streams import Stream


_CONCRETE = {*np.sctypeDict.values()}
# How patos spells the numpy types it names, in messages as in annotations.
SPELLINGS: dict[type, str] = {
    np.int16: "i16", np.int32: "i32", np.int64: "i64", np.uint8: "u8", np.uint16: "u16",
    np.uint32: "u32", np.uint64: "u64", np.float32: "f32", np.float64: "f64",
    np.unsignedinteger: "unsigned", np.number: "number",
}  # fmt: skip


class Shaped(Protocol):
    """An array of any module, host or device, known by its shape and its dtype."""

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def shape(self) -> tuple[int, ...]: ...


@final
class ArrayOf(FrozenModel):
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
class Lanes(FrozenModel):
    """Packed lanes of one 32-bit register, lane 0 in its low bits, each an integer of `element`.

    Device code converts a 32-bit word to lanes and back for free (`u8x4(word)`, `u32(lanes)`),
    and lanes take no arithmetic, so an operation on them is one `primitives.bits` picks for them.
    """

    name: str
    element: type[np.integer]


@final
class ConstantOf(FrozenModel):
    """A record field whose value is part of the record's device type, compiled as a literal."""

    kind: type[int]


if TYPE_CHECKING:
    type Constant[T] = T
else:

    class Constant:
        """A record field whose value is part of the record's device type: `width: Constant[int]`.

        Device code reads it as a literal, so it sizes a local array or unrolls a loop the way a
        closure constant does, and each value compiles its own kernels.
        """

        def __class_getitem__(cls, kind):
            return ConstantOf(kind=kind)


def converted(
    kind: type[np.number] | type[bool], value: int | float | np.number
) -> np.number | bool:
    """`value` as scalar `kind`, refusing an integer's overflow and a float's truncation.

    `OverflowError` or `TypeError` says why it cannot be.
    """
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
            self, index: Subscript | tuple[Subscript, ...] | slice | Shaped, value: Stored
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

        def astype(self, dtype: DTypeLike) -> Numeric[number, *Shape]: ...

        def view(self, dtype: DTypeLike) -> Numeric[number, *Shape]: ...

        def copy(self) -> Self: ...

        def fill(self, value: int) -> None: ...

        def get(self, *, stream: Stream | None = None) -> np.ndarray: ...

    class Packed:
        """What a type checker knows of packed lanes: a 32-bit word converted to and from."""

        def __new__(cls, word: SupportsInt) -> Self: ...

        def __int__(self) -> int: ...

    class u8x4(Packed): ...

    class i8x4(Packed): ...

    class u16x2(Packed): ...

    class i16x2(Packed): ...

    type Vector[T] = Numeric[T, int]
    type Matrix[T] = Numeric[T, int, int]

    # Integers meet as C's do, each converting to every other on assignment, call and return,
    # which no checker models, so a checker reads the scalars as Python's own numbers.
    u8 = u16 = u32 = u64 = i16 = i32 = i64 = unsigned = int
    f32 = f64 = number = float

else:

    def _array(name: str, ndim: int) -> type:
        def __class_getitem__(cls, element):
            if isinstance(element, TypeVar):
                return ArrayOf(element=np.number, ndim=ndim)
            if not (isinstance(element, type) and issubclass(element, np.number)):
                raise TypeError(f"{name}[{element!r}]: an array holds a numeric type")
            return ArrayOf(element=canonical(element), ndim=ndim)

        return type(name, (), {"__class_getitem__": __class_getitem__, "__module__": __name__})

    Vector, Matrix = _array("Vector", 1), _array("Matrix", 2)

    def _numeric(name: str, base: type[np.number]) -> type:
        """`base` as a numeric type: called, its scalar; subscripted, refused as no array.

        A subclass, so numba types a cast through it as one of `base`.
        """

        def __class_getitem__(cls, shape):
            array = "Matrix" if shape == (int, int) else "Vector"
            raise TypeError(f"{name}[...] declares no array; write {array}[{name}]")

        members = {
            "__slots__": (),
            "__new__": lambda cls, value=0: base(value),
            "__class_getitem__": __class_getitem__,
            "__module__": __name__,
        }
        return type(name, (base,), members)

    u8, u16, u32, u64 = (_numeric(f"u{n}", getattr(np, f"uint{n}")) for n in (8, 16, 32, 64))
    i16, i32, i64 = (_numeric(f"i{n}", getattr(np, f"int{n}")) for n in (16, 32, 64))
    f32, f64 = (_numeric(f"f{n}", getattr(np, f"float{n}")) for n in (32, 64))
    unsigned, number = _numeric("unsigned", np.unsignedinteger), _numeric("number", np.number)
    u8x4, i8x4, u16x2, i16x2 = (
        Lanes(name=name, element=element)
        for name, element in [
            ("u8x4", np.uint8),
            ("i8x4", np.int8),
            ("u16x2", np.uint16),
            ("i16x2", np.int16),
        ]
    )


# What arithmetic meets a numeric value with, what an array stores and what indexes it, read
# lazily, so a type checker's names stand in them.
type Operand = int | np.integer | Shaped | Numeric
type Stored = Operand | float | Packed
type Subscript = int | np.integer


def canonical(kind: type[np.number]) -> type[np.number]:
    """The numpy type a numeric type names, which is what patos reads and numba types."""
    return next(
        base for base in kind.__mro__ if issubclass(base, np.number) and base.__module__ == "numpy"
    )
