"""`kernel`: a kernel compiled from annotated Python, launched as
`kernel[items](*arguments, stream=stream)`.

The kernel declares what each item of a launch runs on, a thread, a warp or a block, and how many
threads a block holds, so a launch names only how many items there are; the grid follows, capped
for a kernel that strides over its items. Defined in a `Struct` with an unannotated `self`, it is
a method of the record, launched as `record.kernel[items](*arguments, stream=stream)`.

A launch converts every scalar to the type its parameter declares and hands cuda.core the
arguments marshalled by hand, on the stream it names by the CUDA stream protocol (CuPy's, torch's,
cuda.core's), after waiting for any array whose interface names another stream as its producer.
There is no current stream to fall back on. Arrays are contiguous and scalars arrive at their
declared types, so a kernel compiles once, or once per dtype an open element
(`Vector[unsigned]`) meets; an array of another element, dimension count or layout is refused.

A kernel compiles once for the sources it is made of and the host it runs on. The first process,
cold with an empty cache, compiles its cubin and keeps it in the user's cache directory, and each
later one, warm, loads it (`patos.cuda.artifacts`); an edit to any source its code can reach keys
it anew, so none loads stale, and a process that imported other modules finds it all the same. A
kernel that cannot be kept compiles in every process: one made in a function, one that closes over
values, one whose source is no file, one taking a record class made in a function, and one that
refers to a global device array.
"""

import hashlib
import logging
import sys
from collections.abc import Callable
from enum import StrEnum, auto
from functools import cache, partial
from itertools import chain
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, cast, overload

from cuda.core import Kernel as Compiled
from cuda.core import LaunchConfig, launch
from numba import types

from .. import artifacts
from ..front.declarations import is_scalar, named
from ..front.rewrite import Helper, Items, Rewrite
from ..runtime.streams import borrowed, handle
from ..scalars import ArrayOf, converted
from .arguments import argument
from .compiler import built
from .declared import is_admitted
from .decorators import is_member, made_for, read_bound
from .identity import (
    block_count,
    block_index,
    thread_count,
    thread_index,
    warp_count,
    warp_index,
)

if TYPE_CHECKING:
    from ..runtime.streams import StreamLike
    from .struct import Struct

logger = logging.getLogger(__name__)

# The most blocks a striding kernel takes, many waves on any GPU.
_STRIDED_BLOCKS = 4096


class Per(StrEnum):
    """What one item of a launch runs on."""

    THREAD = auto()
    WARP = auto()
    BLOCK = auto()

    def items(self, *, strided: bool) -> Items:
        """Where this kind of item starts, and how far apart a kernel that strides finds them."""
        first, stride = _IDENTITY[self]
        return Items(first=first, stride=stride if strided else None)

    def lanes(self, threads: int) -> int:
        """How many threads one item takes in a block of `threads`."""
        match self:
            case Per.THREAD:
                return 1
            case Per.WARP:
                return 32
            case _:
                return threads


_IDENTITY: dict[Per, tuple[Helper, Helper]] = {
    Per.THREAD: (thread_index, thread_count),
    Per.WARP: (warp_index, warp_count),
    Per.BLOCK: (block_index, block_count),
}


class Kernel:
    """A kernel compiled from annotated Python, launched over a number of items.

    per: what each item runs on.
    threads: threads per block.
    strided: whether the kernel strides over its items, so a launch caps its grid.
    """

    @overload
    def __get__(self, record: Struct, owner: type) -> Bound: ...

    @overload
    def __get__[Holder](self, record: Holder | None, owner: type[Holder]) -> Kernel: ...

    def __get__[Holder](self, record: Struct | Holder | None, owner: type) -> Kernel | Bound:
        """The kernel itself, or a method bound to the record it is read from."""
        # A member kernel lives only on a record class, so what it is read from is a record.
        return Bound(self, cast("Struct", record)) if self.member and record is not None else self

    def __getitem__(self, items: int) -> Callable[..., None]:
        """The launch over `items` items, taking the arguments and the `stream` to run on."""
        return partial(self.launch, self.grid(items))

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

    def _built(self, kinds: tuple[types.Type, ...], key: str | None) -> artifacts.Cubin:
        """The cubin `kinds` compile to in this process, kept under `key` when there is one."""
        made = built(self.rebuilt, kinds)
        logger.info("compiled %s for %s", self.function.__qualname__, kinds)
        if foreign := made_for(made.library) - {self.threads}:
            raise TypeError(
                f"{self.function.__qualname__} runs {self.threads} threads a block and calls "
                f"device functions made for {', '.join(map(str, sorted(foreign)))}"
            )
        if key is not None and made.keepable:
            artifacts.kept(key, made.cubin)
        return made.cubin

    def _compiled(self, kinds: tuple[types.Type, ...]) -> Compiled:
        """The cuda.core kernel `kinds` compile to, loaded when a process kept it.

        Each array must match its declaration, and every device function made for a block size
        must match the kernel's `threads`; a kernel refused so is never kept.
        """
        for name, kind, given in zip(self.names, self.declared, kinds, strict=True):
            if not isinstance(kind, ArrayOf):
                continue
            admitted = is_admitted(kind, given)
            if admitted and given.layout == "C":
                continue
            held = "strided, not contiguous" if admitted else f"{given}"
            raise TypeError(
                f"{self.function.__qualname__}'s `{name}` is {held}, "
                f"where {named(kind)} is declared"
            )
        key = self._key(kinds)
        cubin = None if key is None else artifacts.loaded(key)
        if cubin is None:
            cubin = self._built(kinds, key)
        compiled = cubin.kernel()
        self.compiled[tuple(map(id, kinds))] = (kinds, compiled)
        return compiled

    def _key(self, kinds: tuple[types.Type, ...]) -> str | None:
        """The digest `kinds` compile under, None for a kernel another process cannot find again.

        That is one made in a function or closing over values, one whose module or source is no
        file, and one taking a record class made in a function.
        """
        function = self.function
        module = sys.modules.get(function.__module__)
        signature = f"{function.__qualname__} {self.per} {self.threads} {self.strided} {kinds}"
        if (
            module is None
            or function.__closure__
            or "<locals>" in signature
            or not Path(function.__code__.co_filename).is_file()
        ):
            return None
        held = f"{artifacts.sources_digest(module)}\0{function.__module__}\0{signature}"
        return hashlib.sha256(held.encode()).hexdigest()[:32]

    def _refuse(self, arguments: tuple) -> None:
        """Raise what converting the first refused scalar raised, naming its parameter."""
        for name, value, convert in zip(self.names, arguments, self.converters, strict=True):
            if convert is None:
                continue
            try:
                convert(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise type(error)(f"{self.function.__qualname__}'s `{name}` {error}") from error

    def bind(self, owner: type[Struct] | None, name: str) -> None:
        """Read and check the kernel, its `self` an `owner` record when it is a method."""
        reading = read_bound(self.function, kernel=True, owner=owner)
        self.rebuilt = Rewrite(reading, self.per.items(strided=self.strided)).rebuilt()
        self.names = tuple(reading.parameters)
        self.declared = tuple(reading.parameters.values())
        self.converters = tuple(
            partial(converted, kind) if is_scalar(kind) or kind is bool else None
            for kind in self.declared
        )

    def grid(self, items: int) -> tuple[int, int]:
        """The blocks and threads that give each item what it runs on."""
        blocks = -(-items * self.per.lanes(self.threads) // self.threads)
        return min(blocks, _STRIDED_BLOCKS) if self.strided else blocks, self.threads

    def launch(self, grid: tuple[int, int], *arguments, stream: StreamLike) -> None:
        """Launch over `grid` on `stream`, which launches nothing when it has no blocks.

        stream: any stream the CUDA stream protocol names (CuPy's, torch's, cuda.core's).
        """
        blocks, threads = grid
        if not blocks:
            return
        number = handle(stream)
        try:
            marshalled = [
                argument(value if convert is None else convert(value), number)
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
        launch(borrowed(number), _config(blocks, threads), kernel, *values)


class Bound:
    """A kernel method of one record, launched with that record as its `self`."""

    __slots__ = ("kernel", "record")

    def __getitem__(self, items: int) -> Callable[..., None]:
        return partial(self.kernel[items], self.record)

    def __init__(self, kernel: Kernel, record: Struct) -> None:
        self.kernel = kernel
        self.record = record


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
