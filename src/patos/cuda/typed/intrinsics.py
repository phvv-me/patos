"""Device functions written as one PTX block, declared by an annotated stub.

The stub's annotations type the call as `device` types a device function: Numba casts every
argument to its declared scalar type, and the checks read the stub's signature. A PTX template
names its output `$result` and its inputs by the stub's parameter names, and each operand's
register constraint follows from its type. A stub returns a scalar, a tuple of scalars (`$result`
is then the vector `{a, b}` of registers the PTX fills, and `$result0`, `$result1` the registers
alone) or `None` for an effect such as a store. A template of register arithmetic alone is pure,
so the compiler may merge or move it; any other, such as a load, a store, an atomic, a barrier, a
warp collective (`*.sync`) or a read of a special register (`%laneid`), runs as an effect where it
is written, and so does an instruction patos does not know.
"""

import inspect
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from string import Template
from types import FunctionType, NoneType
from typing import cast

from llvmlite import ir
from numba import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.cuda.cudadecl import registry as typing_registry
from numba.cuda.cudaimpl import registry as lowering_registry
from numba.cuda.extending import intrinsic

from .declarations import Signature, declared, numba_type
from .lanes import LaneType
from .reading import AnnotationError

# The inline-assembly constraint of the register a PTX operand of each type lives in.
_REGISTERS = {
    types.int16: "h", types.uint16: "h", types.int32: "r", types.uint32: "r",
    types.int64: "l", types.uint64: "l", types.float32: "f", types.float64: "d",
}  # fmt: skip
# The opcodes that compute registers from registers and immediates alone.
_ARITHMETIC = frozenset({
    "abs", "add", "and", "bfe", "bfi", "bfind", "brev", "clz", "cvt", "dp2a", "dp4a", "lop3",
    "mad", "max", "min", "mov", "mul", "neg", "not", "or", "popc", "prmt", "sad", "selp", "setp",
    "shf", "shl", "shr", "sub", "vabsdiff2", "vabsdiff4", "xor",
})  # fmt: skip
# A statement's optional label and predicate guard, and the opcode they precede.
_STATEMENT = re.compile(r"^(?:\w+:\s*)?(?:@!?\w+\s+)?([\w.]+)")


def ptx[F: FunctionType](template: str) -> Callable[[F], F]:
    """Make an annotated stub a device function running the PTX `template`.

    template: PTX naming the output `$result` and each input `$<parameter>`.
    """

    def defined(stub: F) -> F:
        signature, typed = read_stub(stub)
        outputs = _outputs(typed.return_type)
        constraints = _constraints(stub, outputs, typed)
        text = Template(template).substitute(_operands(stub, len(outputs)))
        effect = not (outputs and is_pure(template))

        def lowered(context, builder: ir.IRBuilder, call, arguments: list[ir.Value]) -> ir.Value:
            assembly = ir.InlineAsm(
                _function_type(context, outputs, call), text, constraints, side_effect=effect
            )
            value = builder.call(assembly, arguments)
            if isinstance(call.return_type, types.BaseTuple):
                parts = [builder.extract_value(value, index) for index in range(len(outputs))]
                return context.make_tuple(builder, call.return_type, parts)
            return value if outputs else context.get_dummy_value()

        compiled = as_intrinsic(stub, signature, typed, lowered)
        compiled.__dict__["template"] = template
        return compiled

    return defined


def is_pure(template: str) -> bool:
    """Whether every instruction of the PTX `template` is register arithmetic.

    Each is an opcode of `_ARITHMETIC` over registers and immediates, reading no special register
    and writing no carry (`.cc`, which a later `addc` reads), so the compiler may merge or move it.
    """
    found = [_STATEMENT.match(statement) for statement in _statements(template)]
    return "%" not in template and all(
        match is not None
        and (opcode := match[1].split("."))[0] in _ARITHMETIC
        and "cc" not in opcode
        for match in found
    )


def _statements(template: str) -> Iterator[str]:
    """Each instruction of the PTX `template`, as one line, with directives and comments left out.

    A directive such as `.reg` or `.loc` ends at its line, not at a `;`, so it is dropped by line
    from the chunk that holds the instruction behind it.
    """
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", template, flags=re.DOTALL)
    for chunk in code.split(";"):
        lines = (line.strip(" \t{}") for line in chunk.splitlines())
        if statement := " ".join(line for line in lines if line and not line.startswith(".")):
            yield statement


def read_stub(stub: FunctionType) -> tuple[Signature, templates.Signature]:
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
    return Signature(parameters=tuple(kinds), returns=returns), result(*typed)


def _outputs(returns: types.Type) -> list[types.Type]:
    """The registers a PTX block writes: a tuple's elements, a scalar, or none for `None`."""
    if isinstance(returns, types.BaseTuple):
        return list(returns.types)
    return [] if returns == types.none else [returns]


def _constraints(
    stub: FunctionType, outputs: Sequence[types.Type], typed: templates.Signature
) -> str:
    """The inline-assembly constraints of the operands, the written registers first."""
    try:
        return ",".join(
            [f"={_register(kind)}" for kind in outputs] + list(map(_register, typed.args))
        )
    except KeyError:
        raise AnnotationError(
            f"{stub.__qualname__}: a PTX operand is 16, 32 or 64 bits wide"
        ) from None


def _register(kind: types.Type) -> str:
    """The constraint of the register an operand of `kind` lives in; lanes fill a 32-bit one."""
    return "r" if isinstance(kind, LaneType) else _REGISTERS[kind]


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


def as_intrinsic[F: FunctionType](
    stub: F, signature: Signature, typed: templates.Signature, lowered: Callable[..., ir.Value]
) -> F:
    """A Numba intrinsic typed by the stub's signature, carrying it for the callers' checks."""

    @_named_like(stub)
    def typing(_context, *_arguments: types.Type) -> tuple:
        return typed, lowered

    function = intrinsic(typing)
    function.__dict__["device_signature"] = signature
    return cast("F", function)


def as_dispatching[F: FunctionType](
    stub: F, resolve: Callable[..., tuple[templates.Signature, Callable[..., ir.Value]]]
) -> F:
    """The stub as a device function that types and lowers a call as `resolve` decides.

    A call names its operands as the stub does, by position or by keyword. Numba asks about an
    int literal before its plain `int64`, and a call refused that way is refused for good: asked
    again with `int64`, a tie between the integer types the literal fits would resolve to one.
    """
    parameters = inspect.signature(stub)
    lowerings: dict[tuple[types.Type, ...], Callable[..., ir.Value]] = {}

    class Dispatching(templates.AbstractTemplate):
        prefer_literal = True
        refusal: TypingError | None = None

        def generic(self, args: tuple[types.Type, ...], kws: Mapping[str, types.Type]):
            if self.refusal is None:
                try:
                    return self._typed(args, kws)
                except TypingError as error:
                    self.refusal = error
            raise self.refusal

        def _typed(self, args, kws) -> templates.Signature:
            try:
                bound = parameters.bind(*args, **kws)
            except TypeError as error:
                raise TypingError(f"{stub.__qualname__}: {error}") from None
            typed, lowered = resolve(self.context, bound.args)
            lowerings[typed.args] = lowered
            return typed.replace(pysig=parameters)

    # The stub is no module attribute yet, so the template is what a call is typed and lowered by.
    @lowering_registry.lower(Dispatching, types.VarArg(types.Any))
    def lowered(context, builder: ir.IRBuilder, call: templates.Signature, values: Sequence):
        return lowerings[call.args](context, builder, call, values)

    typing_registry.register_global(stub, typing_key=Dispatching)(Dispatching)
    return stub


def _named_like(stub: FunctionType) -> Callable[[FunctionType], FunctionType]:
    """A decorator naming and documenting a typing function as the stub, taking its parameters.

    Numba folds a call's arguments by the signature the typing function carries, after the
    typing context.
    """
    context = inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    parameters = [context, *inspect.signature(stub).parameters.values()]

    def named(typing: FunctionType) -> FunctionType:
        typing.__dict__["__signature__"] = inspect.Signature(parameters)
        typing.__name__, typing.__qualname__, typing.__doc__ = (
            stub.__name__,
            stub.__qualname__,
            stub.__doc__,
        )
        return typing

    return named
