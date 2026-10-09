"""`device`: numba-cuda's device `cuda.jit`, reading the annotations first.

A device function compiles once at the parameter types its annotations declare, and numba-cuda
casts every argument at the call. NVVM inlines it into its callers, which compiles a
cold process about three times faster than Numba re-typing its body inside every caller.

Defined in a `Struct` with an unannotated `self`, it is a device member of the record instead:
a method, an operator when it is a dunder (`__getitem__`, `__len__`, `__contains__`), or an
attribute under `@property`. Defined so in a named value's `typing.NamedTuple`, it is a method or
an attribute of the value, compiled when device code first names the class.
"""

import annotationlib
import ast
from collections.abc import Callable, Iterable, Iterator
from types import FunctionType
from typing import TYPE_CHECKING, cast, overload
from weakref import WeakKeyDictionary, WeakSet

from numba import cuda, types
from numba.core.errors import TypingError
from numba.cuda.codegen import CUDACodeLibrary
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda.np.numpy_support import as_dtype

from ..scalars import ArrayOf
from .checks import read
from .declarations import Declared, NamedValue, Returns, named, numba_type
from .reading import Reading
from .records import register_member
from .rewrite import Rewrite
from .values import register_value_member

if TYPE_CHECKING:
    from .struct import Struct

# The block size each compiled `threads` device function was made for, by the library it became.
_MADE_FOR: WeakKeyDictionary[CUDACodeLibrary, int] = WeakKeyDictionary()
# The named value classes whose device members are compiled, and those compiling, which the
# members' own annotations may name again. A class whose members failed is neither.
_BOUND: WeakSet[type] = WeakSet()
_BINDING: WeakSet[type] = WeakSet()


# `device` returns the numba dispatcher typed as the function it compiles, so a device function
# calling another reads the callee's own signature.
@overload
def device[F: FunctionType](
    function: F, *, inline: bool = True, threads: int | None = None
) -> F: ...


@overload
def device[F: FunctionType](
    function: None = None, *, inline: bool = True, threads: int | None = None
) -> Callable[[F], F]: ...


def device[F: FunctionType](
    function: F | None = None, *, inline: bool = True, threads: int | None = None
) -> F | Callable[[F], F]:
    """Compile `function` as a device function whose annotations convert what they name.

    inline: let NVVM inline it into its callers; `@device(inline=False)` keeps one out-of-line
        copy a large caller calls into. A function without parameters is instead inlined into
        the IR of its callers, which then compile to what they would with its expression written
        in place.
    threads: the block size of the only kernels that can call a function made for it, such as a
        block reduction sized to its warps; a kernel of another size refuses to compile it in.
    """

    def compiled(defined: F) -> F:
        if is_member(defined):
            return cast("F", Method(defined, inline=inline, threads=threads))
        reading = read_bound(defined, kernel=False)
        return cast("F", _Declared(reading, inline=inline, threads=threads))

    return compiled if function is None else compiled(function)


def read_bound(function: FunctionType, *, kernel: bool, owner: type | None = None) -> Reading:
    """`function` read and checked, the members of every named value it names compiled first.

    owner: the record or named value class `function` is a member of.
    """
    reading = read(function, kernel=kernel, owner=owner)
    built = [reading.constructed(node) for node in reading.walk() if isinstance(node, ast.Call)]
    classes = set(_named_classes([*reading.declared.values(), reading.returns, *built]))
    for cls in classes - set(_BOUND) - set(_BINDING):
        _BINDING.add(cls)
        try:
            bind_members(cls)
        finally:
            _BINDING.discard(cls)
        _BOUND.add(cls)
    return reading


def bind_members(owner: type) -> None:
    """Compile the device members `owner` defines, its methods and `@property` attributes."""
    for name, member in list(vars(owner).items()):
        match member:
            case property(fget=Method() as method):
                method.bind(owner, name, attribute=True)
            case Method():
                member.bind(owner, name)


def is_member(function: FunctionType) -> bool:
    """Whether `function` is a member of its class: its first parameter an unannotated `self`."""
    names = function.__code__.co_varnames[: function.__code__.co_argcount]
    annotations = annotationlib.get_annotations(function, format=annotationlib.Format.FORWARDREF)
    return bool(names) and names[0] == "self" and "self" not in annotations


class Method:
    """A device function defined in a record or named value class, compiled once bound to it."""

    def __init__(self, function: FunctionType, *, inline: bool, threads: int | None) -> None:
        self.function = function
        self.inline = inline
        self.threads = threads

    def bind(
        self, owner: type[Struct] | type[tuple], name: str, *, attribute: bool = False
    ) -> None:
        """Compile with `self` typed as an `owner` and register it on the device type of `owner`.

        A device member lives on the device only, so the host class loses it.
        """
        reading = read_bound(self.function, kernel=False, owner=owner)
        declared = _Declared(reading, inline=self.inline, threads=self.threads)
        if issubclass(owner, tuple):
            register_value_member(owner, name, declared, attribute=attribute)
        else:
            register_member(owner, name, declared, attribute=attribute)
        delattr(owner, name)


class _Declared(CUDADispatcher):
    """A device function compiled at the parameter types its annotations declare, whatever a
    caller passes.

    A scalar parameter compiles at its type, which Numba casts every argument to at the call. An
    array parameter compiles in the layout the caller passes, and refuses an argument of another
    element or dimension count. A named value, which Numba converts to no other, is refused unless
    its class and every field are the declared ones. A record or a type parameter keeps the type
    the caller gives it. A function without parameters has no types to compile at, so Numba
    inlines it into the IR of its callers, which hold the expression it returns as if they had
    written it.
    """

    def __init__(self, reading: Reading, *, inline: bool, threads: int | None) -> None:
        rebuilt = Rewrite(reading).rebuilt()
        options = cuda.jit(device=True)(rebuilt).targetoptions | {"forceinline": inline}
        if inline and not reading.parameters:
            options["inline"] = "always"
        super().__init__(rebuilt, targetoptions=options)
        self.declared = reading.parameters
        self.threads = threads

    def compile_device(self, args: tuple[types.Type, ...], return_type: types.Type | None = None):
        declared = tuple(
            self._parameter(name, kind, given)
            for (name, kind), given in zip(self.declared.items(), args, strict=True)
        )
        compiled = super().compile_device(declared, return_type)
        if self.threads is not None:
            _MADE_FOR[compiled.library] = self.threads
        return compiled

    def _parameter(self, name: str, kind: Declared, given: types.Type) -> types.Type:
        """The type parameter `name` compiles at, when a caller passes `given`."""
        if not isinstance(kind, ArrayOf | NamedValue):
            return numba_type(kind) or given
        if not _is_admitted(kind, given):
            raise TypingError(
                f"{self.py_func.__qualname__}'s `{name}` receives {given} where {named(kind)} "
                "is declared"
            )
        return given


def _named_classes(kinds: Iterable[Returns | None]) -> Iterator[type[tuple]]:
    """The named value classes `kinds` hold, in their fields and tuples as well."""
    for kind in kinds:
        if isinstance(kind, NamedValue):
            yield kind.cls
            yield from _named_classes(kind.kinds())
        elif isinstance(kind, tuple):
            yield from _named_classes(kind)


def made_for(library: CUDACodeLibrary) -> set[int]:
    """The block sizes of the `threads` device functions linked into `library`, however deep."""
    return {_MADE_FOR[linked] for linked in library.linking_libraries if linked in _MADE_FOR}


def _is_admitted(kind: Declared, given: types.Type) -> bool:
    """Whether `given` is a value of the array or named value `kind` declares, field by field."""
    if isinstance(kind, ArrayOf):
        return isinstance(given, types.Array) and kind.admits(as_dtype(given.dtype), given.ndim)
    if isinstance(kind, NamedValue):
        return (
            isinstance(given, types.BaseNamedTuple)
            and given.instance_class is kind.cls
            and len(given) == len(kind.fields)
            and all(
                _is_admitted(field, held) for field, held in zip(kind.kinds(), given, strict=True)
            )
        )
    expected = numba_type(kind)
    return expected is None or types.unliteral(given) == expected
