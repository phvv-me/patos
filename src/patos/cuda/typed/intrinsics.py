"""Device functions written as one PTX block, declared by an annotated stub.

The stub's annotations type the call as `device` types a device function: Numba casts every
argument to its declared scalar type, and the checks read the stub's signature. A PTX template
names its output `$result` and its inputs by the stub's parameter names, and each operand's
register constraint follows from its type. Warp collectives stay where they are written, since a
template runs as an effect unless declared `pure`.
"""

import inspect
from collections.abc import Callable
from string import Template
from types import FunctionType
from typing import cast

from llvmlite import ir
from numba import types
from numba.core.typing import templates
from numba.cuda.extending import intrinsic

from .declarations import Signature, declared, numba_type
from .reading import AnnotationError

# The inline-assembly constraint of the register a PTX operand of each type lives in.
_REGISTERS = {
    types.int16: "h", types.uint16: "h", types.int32: "r", types.uint32: "r",
    types.int64: "l", types.uint64: "l", types.float32: "f", types.float64: "d",
}  # fmt: skip


def ptx[F: FunctionType](template: str, *, pure: bool = False) -> Callable[[F], F]:
    """Make an annotated stub a device function running the PTX `template`.

    template: PTX naming the output `$result` and each input `$<parameter>`.
    pure: the PTX only computes its output from its inputs, so the compiler may merge or move it;
        a warp collective is never pure.
    """

    def defined(stub: F) -> F:
        signature, typed = _declared_signature(stub)
        names = ["result", *inspect.signature(stub).parameters]
        text = Template(template).substitute(
            {name: f"${index}" for index, name in enumerate(names)}
        )
        operands = [typed.return_type, *typed.args]
        if any(kind not in _REGISTERS for kind in operands):
            raise AnnotationError(f"{stub.__qualname__}: a PTX operand is 16, 32 or 64 bits wide")
        registers = [_REGISTERS[kind] for kind in operands]
        constraints = ",".join([f"={registers[0]}", *registers[1:]])

        def lowered(context, builder: ir.IRBuilder, call, arguments: list[ir.Value]) -> ir.Value:
            assembly = ir.InlineAsm(
                _function_type(context, call), text, constraints, side_effect=not pure
            )
            return builder.call(assembly, arguments)

        return _intrinsic(stub, signature, typed, lowered)

    return defined


def _declared_signature(stub: FunctionType) -> tuple[Signature, templates.Signature]:
    """What the stub declares, every parameter and the return a scalar, and its Numba signature."""
    annotations = inspect.get_annotations(stub)
    names = [*inspect.signature(stub).parameters, "return"]
    kinds = [declared(annotations.get(name)) for name in names]
    typed = [kind for kind in map(numba_type, kinds) if kind is not None]
    if len(typed) < len(names):
        raise AnnotationError(f"{stub.__qualname__}: a parameter or the return declares no scalar")
    *parameters, returns = typed
    return Signature(tuple(kinds[:-1]), kinds[-1]), returns(*parameters)


def _function_type(context, call: templates.Signature) -> ir.FunctionType:
    """The LLVM function type of a call Numba typed as `call`."""
    return ir.FunctionType(
        context.get_value_type(call.return_type),
        [context.get_value_type(kind) for kind in call.args],
    )


def _intrinsic[F: FunctionType](
    stub: F, signature: Signature, typed: templates.Signature, lowered: Callable[..., ir.Value]
) -> F:
    """A Numba intrinsic typed by the stub's signature, carrying it for the callers' checks."""

    def typing(_context, *_arguments: types.Type) -> tuple:
        return typed, lowered

    # Numba folds a call's arguments by this signature: the typing context, then the stub's own.
    context = inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    typing.__dict__["__signature__"] = inspect.Signature(
        [context, *inspect.signature(stub).parameters.values()]
    )
    typing.__name__, typing.__qualname__, typing.__doc__ = (
        stub.__name__,
        stub.__qualname__,
        stub.__doc__,
    )
    function = intrinsic(typing)
    function.__dict__["device_signature"] = signature
    return cast("F", function)
