"""`device`: numba-cuda's device `cuda.jit`, reading the annotations first.

A device function compiles once at the parameter types its annotations declare, and numba-cuda
casts every argument at the call. NVVM inlines it into its callers, which compiles a
cold process about three times faster than Numba re-typing its body inside every caller.

Defined in a `Struct` with an unannotated `self`, it is a device member of the record instead:
a method, an operator when it is a dunder (`__getitem__`, `__len__`, `__contains__`), or an
attribute under `@property`.
"""

import annotationlib
from collections.abc import Callable
from types import FunctionType
from typing import TYPE_CHECKING, cast, overload

from numba import cuda, types
from numba.core.errors import TypingError
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda.np.numpy_support import as_dtype

from .checks import read
from .declarations import Declared, named, numba_type
from .reading import Reading
from .records import register_member
from .rewrite import Rewrite
from .scalars import ArrayOf

if TYPE_CHECKING:
    from .struct import Struct


# `device` returns the numba dispatcher typed as the function it compiles, so a device function
# calling another reads the callee's own signature.
@overload
def device[F: FunctionType](function: F, *, inline: bool = True) -> F: ...


@overload
def device[F: FunctionType](function: None = None, *, inline: bool = True) -> Callable[[F], F]: ...


def device[F: FunctionType](
    function: F | None = None, *, inline: bool = True
) -> F | Callable[[F], F]:
    """Compile `function` as a device function whose annotations convert what they name.

    inline: let NVVM inline it into its callers; `@device(inline=False)` keeps one out-of-line
        copy a large caller calls into.
    """

    def compiled(defined: F) -> F:
        if is_member(defined):
            return cast("F", Method(defined, inline=inline))
        return cast("F", _Declared(read(defined, kernel=False), inline=inline))

    return compiled if function is None else compiled(function)


def is_member(function: FunctionType) -> bool:
    """Whether `function` is a record's member: its first parameter an unannotated `self`."""
    names = function.__code__.co_varnames[: function.__code__.co_argcount]
    annotations = annotationlib.get_annotations(function, format=annotationlib.Format.FORWARDREF)
    return bool(names) and names[0] == "self" and "self" not in annotations


class Method:
    """A device function defined in a record class, compiled once the class exists."""

    def __init__(self, function: FunctionType, *, inline: bool) -> None:
        self.function = function
        self.inline = inline

    def bind(self, owner: type[Struct], name: str, *, attribute: bool = False) -> None:
        """Compile with `self` typed as an `owner` record and register it on that record's type.

        A device member lives on the device only, so the host class loses it.
        """
        declared = _Declared(read(self.function, kernel=False, owner=owner), inline=self.inline)
        register_member(owner, name, declared, attribute=attribute)
        delattr(owner, name)


class _Declared(CUDADispatcher):
    """A device function compiled at the parameter types its annotations declare, whatever a
    caller passes.

    A scalar parameter compiles at its type, which Numba casts every argument to at the call. An
    array parameter compiles in the layout the caller passes, and refuses an argument of another
    element or dimension count. A record or a type parameter keeps the type the caller gives it.
    """

    def __init__(self, read: Reading, *, inline: bool) -> None:
        rebuilt = Rewrite(read).rebuilt()
        options = cuda.jit(device=True)(rebuilt).targetoptions | {"forceinline": inline}
        super().__init__(rebuilt, targetoptions=options)
        self.declared = read.parameters

    def compile_device(self, args: tuple[types.Type, ...], return_type: types.Type | None = None):
        declared = tuple(
            self._parameter(name, kind, given)
            for (name, kind), given in zip(self.declared.items(), args, strict=True)
        )
        return super().compile_device(declared, return_type)

    def _parameter(self, name: str, kind: Declared, given: types.Type) -> types.Type:
        """The type parameter `name` compiles at, when a caller passes `given`."""
        if not isinstance(kind, ArrayOf):
            return numba_type(kind) or given
        if not isinstance(given, types.Array) or not kind.admits(
            as_dtype(given.dtype), given.ndim
        ):
            raise TypingError(
                f"{self.py_func.__qualname__}'s `{name}` receives {given} where {named(kind)} "
                "is declared"
            )
        return given
