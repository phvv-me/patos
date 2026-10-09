"""Device functions written as one PTX block, declared by an annotated stub.

The stub's annotations type the call as `device` types a device function: Numba casts every
argument to its declared scalar type, and the checks read the stub's signature. A PTX template
names its output `$result` and its inputs by the stub's parameter names, and each operand's
register constraint follows from its type. A stub returns a scalar, a tuple of scalars (`$result`
is then the vector `{a, b}` of registers the PTX fills, and `$result0`, `$result1` the registers
alone) or `None` for an effect such as a store. Warp collectives stay where they are written,
since a template runs as an effect unless declared `pure`.
"""

import inspect
from collections.abc import Callable, Sequence
from string import Template
from types import FunctionType, NoneType
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
        outputs = _outputs(typed.return_type)
        constraints = _constraints(stub, outputs, typed)
        text = Template(template).substitute(_operands(stub, len(outputs)))

        def lowered(context, builder: ir.IRBuilder, call, arguments: list[ir.Value]) -> ir.Value:
            assembly = ir.InlineAsm(
                _function_type(context, outputs, call), text, constraints, side_effect=not pure
            )
            value = builder.call(assembly, arguments)
            if isinstance(call.return_type, types.BaseTuple):
                parts = [builder.extract_value(value, index) for index in range(len(outputs))]
                return context.make_tuple(builder, call.return_type, parts)
            return value if outputs else context.get_dummy_value()

        return _intrinsic(stub, signature, typed, lowered)

    return defined


def _declared_signature(stub: FunctionType) -> tuple[Signature, templates.Signature]:
    """Read what the stub declares and the Numba signature that types a call of it.

    Every parameter is a scalar and the return a scalar, a tuple of them or None.
    """
    read = inspect.signature(stub)
    kinds = [declared(parameter.annotation) for parameter in read.parameters.values()]
    returns = NoneType if read.return_annotation is None else declared(read.return_annotation)
    typed = [numba_type(kind) for kind in kinds]
    result = types.none if returns is NoneType else numba_type(returns)
    if result is None or None in typed:
        raise AnnotationError(
            f"{stub.__qualname__}: a parameter or the return declares no scalar, tuple or None"
        )
    return Signature(tuple(kinds), returns), result(*typed)


def _outputs(returns: types.Type) -> list[types.Type]:
    """The registers a PTX block writes: a tuple's elements, a scalar, or none for `None`."""
    if isinstance(returns, types.BaseTuple):
        return list(returns.types)
    return [] if returns == types.none else [returns]


def _constraints(
    stub: FunctionType, outputs: Sequence[types.Type], typed: templates.Signature
) -> str:
    """The inline-assembly constraints of the operands, the written registers first."""
    if any(kind not in _REGISTERS for kind in (*outputs, *typed.args)):
        raise AnnotationError(f"{stub.__qualname__}: a PTX operand is 16, 32 or 64 bits wide")
    written = [f"={_REGISTERS[kind]}" for kind in outputs]
    return ",".join(written + [_REGISTERS[kind] for kind in typed.args])


def _operands(stub: FunctionType, written: int) -> dict[str, str]:
    """Spell the operands a template names, `written` registers first and then the parameters.

    `$result` is the one register written or the vector of them, `$result<i>` register `i`.
    """
    registers = [f"${index}" for index in range(written)]
    spelled = {"result": f"{{{', '.join(registers)}}}" if written > 1 else "".join(registers)}
    spelled |= {f"result{index}": register for index, register in enumerate(registers)}
    parameters = inspect.signature(stub).parameters
    return spelled | {name: f"${written + index}" for index, name in enumerate(parameters)}


def _function_type(
    context, outputs: Sequence[types.Type], call: templates.Signature
) -> ir.FunctionType:
    """The LLVM function type of the assembly a call Numba typed as `call` runs.

    It returns a struct of the registers it writes, which Numba packs into the tuple declared.
    """
    written = [context.get_value_type(kind) for kind in outputs]
    match written:
        case []:
            returns = ir.VoidType()
        case [only]:
            returns = only
        case _:
            returns = ir.LiteralStructType(written)
    return ir.FunctionType(returns, [context.get_value_type(kind) for kind in call.args])


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
