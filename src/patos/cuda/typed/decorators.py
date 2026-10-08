"""`device` and `kernel`: numba-cuda's `cuda.jit`, reading the annotations first.

A device function compiles once at the parameter types its annotations declare, and numba-cuda
casts every argument at the call. NVVM inlines it into its callers, which compiles a
cold process about three times faster than Numba re-typing its body inside every caller.
"""

from collections.abc import Callable
from types import FunctionType
from typing import cast, overload

from numba import cuda, types
from numba.cuda.dispatcher import CUDADispatcher

from .checks import Checks
from .declarations import numba_type
from .inference import Inference
from .reading import Function
from .rewrite import Rewrite


# `device` and `kernel` return the numba dispatcher typed as the function it compiles, so a device
# function calling another reads the callee's own signature.
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
        return cast("F", _Declared(_read(defined, kernel=False), inline=inline))

    return compiled if function is None else compiled(function)


def kernel[F: FunctionType](function: F) -> F:
    """Compile `function` as a kernel whose annotations convert what they name."""
    return cuda.jit(Rewrite(_read(function, kernel=True)).rebuilt())


class _Declared(CUDADispatcher):
    """A device function compiled at the parameter types its annotations declare, whatever a
    caller passes.

    A parameter declaring no scalar (an array, a record, a type parameter) keeps the type the
    caller gives it.
    """

    def __init__(self, read: Function, *, inline: bool) -> None:
        rebuilt = Rewrite(read).rebuilt()
        options = cuda.jit(device=True)(rebuilt).targetoptions | {"forceinline": inline}
        super().__init__(rebuilt, targetoptions=options)
        self.parameters = [numba_type(kind) for kind in read.parameters.values()]

    def compile_device(self, args: tuple[types.Type, ...], return_type: types.Type | None = None):
        declared = tuple(
            given if kind is None else kind
            for kind, given in zip(self.parameters, args, strict=True)
        )
        return super().compile_device(declared, return_type)


def _read(function: FunctionType, *, kernel: bool) -> Function:
    """`function` read and checked.

    Raises `AnnotationError` naming every line whose annotations are missing, contradict each
    other, or are repeated by a cast.
    """
    read = Function(function, kernel=kernel)
    Checks(read, Inference(read)).run()
    return read
