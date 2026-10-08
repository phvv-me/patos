"""`kernel`: a kernel compiled from annotated Python and launched as `kernel[items](*arguments)`.

The kernel declares what each item of a launch runs on, a thread, a warp or a block, and how many
threads a block holds, so a launch names only how many items there are; the grid follows, capped
for a kernel that strides over its items. Defined in a `Struct` with an unannotated `self`, it is
a method of the record, launched as `record.kernel[items](*arguments)`.

A launch converts every scalar to the type its parameter declares and hands cuda.core the
arguments marshalled by hand, on the current CuPy stream. Arrays are contiguous and scalars arrive
at their declared types, so a kernel compiles once, or once per dtype an open element
(`unsigned[int]`) meets; an array of another element, dimension count or layout is refused.
"""

from collections.abc import Callable
from enum import StrEnum, auto
from functools import cache, partial
from itertools import chain
from types import FunctionType
from typing import TYPE_CHECKING, cast, overload

from cuda.core import Kernel as Compiled
from cuda.core import LaunchConfig, Stream, launch
from cupy.cuda import get_current_stream
from numba import cuda, types
from numba.cuda.np.numpy_support import as_dtype

from .arguments import argument
from .checks import read
from .declarations import is_scalar, named
from .decorators import is_member
from .rewrite import Rewrite
from .scalars import ArrayOf, converted

if TYPE_CHECKING:
    from .struct import Struct

_WARP = 32
# The most blocks a striding kernel takes, many waves on any GPU.
_STRIDED_BLOCKS = 4096


class Per(StrEnum):
    """What one item of a launch runs on."""

    THREAD = auto()
    WARP = auto()
    BLOCK = auto()

    def lanes(self, threads: int) -> int:
        """How many threads one item takes in a block of `threads`."""
        match self:
            case Per.THREAD:
                return 1
            case Per.WARP:
                return _WARP
        return threads


class Kernel:
    """A kernel compiled from annotated Python, launched over a number of items.

    per: what each item runs on.
    threads: threads per block.
    strided: whether the kernel strides over its items, so a launch caps its grid.
    """

    def __init__(self, function: FunctionType, *, per: Per, threads: int, strided: bool) -> None:
        self.function = function
        self.per = per
        self.threads = threads
        self.strided = strided
        self.member = is_member(function)
        self.compiled: dict[tuple[int, ...], tuple[tuple[types.Type, ...], Compiled]] = {}
        if not self.member:
            self.bind(None, function.__name__)

    def __set_name__(self, owner: type, name: str) -> None:
        if self.member and not hasattr(owner, "__record_fields__"):
            raise TypeError(f"kernel {name} takes `self`, which only a Struct gives it")

    def bind(self, owner: type[Struct] | None, name: str) -> None:
        """Read and check the kernel, its `self` an `owner` record when it is a method."""
        reading = read(self.function, kernel=True, owner=owner)
        self.dispatcher = cuda.jit(Rewrite(reading).rebuilt())
        self.names = tuple(reading.parameters)
        self.declared = tuple(reading.parameters.values())
        self.converters = tuple(
            partial(converted, kind) if is_scalar(kind) or kind is bool else None
            for kind in self.declared
        )

    @overload
    def __get__(self, record: Struct, owner: type) -> Bound: ...

    @overload
    def __get__[Holder](self, record: Holder | None, owner: type[Holder]) -> Kernel: ...

    def __get__[Holder](self, record: Struct | Holder | None, owner: type) -> Kernel | Bound:
        """The kernel itself, or a method bound to the record it is read from."""
        # A member kernel lives only on a record class, so what it is read from is a record.
        return Bound(self, cast("Struct", record)) if self.member and record is not None else self

    def __getitem__(self, items: int) -> Callable[..., None]:
        """The launch over `items` items on the current CuPy stream."""
        return partial(self.launch, self.grid(items))

    def grid(self, items: int) -> tuple[int, int]:
        """The blocks and threads that give each item what it runs on."""
        blocks = -(-items * self.per.lanes(self.threads) // self.threads)
        return min(blocks, _STRIDED_BLOCKS) if self.strided else blocks, self.threads

    def launch(self, grid: tuple[int, int], *arguments) -> None:
        """Launch over `grid`, which launches nothing when it has no blocks."""
        blocks, threads = grid
        if not blocks:
            return
        try:
            marshalled = [
                argument(value if convert is None else convert(value))
                for value, convert in zip(arguments, self.converters, strict=True)
            ]
        except TypeError, ValueError, OverflowError:
            self._refuse(arguments)
            raise
        kinds = tuple(kind for kind, _ in marshalled)
        # Numba interns its types and the cache keeps the ones a key names, so identity tells
        # signatures apart without hashing them.
        held = self.compiled.get(tuple(map(id, kinds)))
        kernel = self._compiled(kinds) if held is None else held[1]
        values = chain.from_iterable(values for _, values in marshalled)
        launch(_stream(get_current_stream().ptr), _config(blocks, threads), kernel, *values)

    def _refuse(self, arguments: tuple) -> None:
        """Raise what converting the first refused scalar raised, naming its parameter."""
        for name, value, convert in zip(self.names, arguments, self.converters, strict=True):
            if convert is None:
                continue
            try:
                convert(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise type(error)(f"{self.function.__qualname__}'s `{name}` {error}") from error

    def _compiled(self, kinds: tuple[types.Type, ...]) -> Compiled:
        """The cuda.core kernel `kinds` compile to, once each array matches its declaration."""
        for name, kind, given in zip(self.names, self.declared, kinds, strict=True):
            if not isinstance(kind, ArrayOf):
                continue
            admitted = isinstance(given, types.Array) and kind.admits(
                as_dtype(given.dtype), given.ndim
            )
            if admitted and given.layout == "C":
                continue
            held = "strided, not contiguous" if admitted else f"{given}"
            raise TypeError(
                f"{self.function.__qualname__}'s `{name}` is {held}, "
                f"where {named(kind)} is declared"
            )
        # numba-cuda 0.30 keeps no public handle to a compiled kernel's cuda.core kernel.
        compiled = self.dispatcher.compile(kinds)._codelibrary.get_cufunc().kernel
        self.compiled[tuple(map(id, kinds))] = (kinds, compiled)
        return compiled


class Bound:
    """A kernel method of one record, launched with that record as its `self`."""

    __slots__ = ("kernel", "record")

    def __init__(self, kernel: Kernel, record: Struct) -> None:
        self.kernel = kernel
        self.record = record

    def __getitem__(self, items: int) -> Callable[..., None]:
        return partial(self.kernel[items], self.record)


@overload
def kernel(function: FunctionType) -> Kernel: ...


@overload
def kernel(
    *, per: Per = Per.THREAD, threads: int = 128, strided: bool = False
) -> Callable[[FunctionType], Kernel]: ...


def kernel(
    function: FunctionType | None = None,
    *,
    per: Per = Per.THREAD,
    threads: int = 128,
    strided: bool = False,
) -> Kernel | Callable[[FunctionType], Kernel]:
    """Compile `function` as a kernel whose annotations convert what they name.

    per: what each item of a launch runs on, a thread by default.
    threads: threads per block.
    strided: whether the kernel strides over its items, so a launch caps its grid.
    """
    compiled = partial(Kernel, per=per, threads=threads, strided=strided)
    return compiled if function is None else compiled(function)


@cache
def _config(blocks: int, threads: int) -> LaunchConfig:
    return LaunchConfig(grid=blocks, block=threads)


@cache
def _stream(handle: int) -> Stream:
    """The CuPy stream `handle` names, borrowed as a cuda.core stream once."""
    return Stream.from_handle(handle)
