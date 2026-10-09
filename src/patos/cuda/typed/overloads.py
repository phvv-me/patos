"""One device function name over several typed implementations, as C++ overloads a name, and the
calls of a device function that a choice forwards to."""

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import FunctionType

import numpy as np
from llvmlite import ir
from numba import types
from numba.core.errors import TypingError
from numba.core.typing import templates

from .declarations import named, numba_type
from .intrinsics import as_dispatching
from .signatures import Kinds, implementing, listed, taken


def dispatched[F: FunctionType](*implementations: Callable) -> Callable[[F], F]:
    """Make a stub one device function over typed `implementations`, as C++ overloads a name.

    The stub declares what it takes as a type checker reads it: each `typing.overload` of it, or
    its own signature, where a constrained type parameter stands for each of its constraints, all
    the parameters of one standing for the same. A signature runs the implementation that declares
    its types, or else the one generic over them with those constraints, and returns what the
    stub declares; every implementation runs one, and every overload takes the stub's arity.
    Anything else is refused at import (`signatures`).

    A call runs the signature its operands fit best, picked at compile time. One fits when each
    integer operand converts to an integer parameter and every other operand is the type its
    parameter declares; of those, the one declaring the most operands' own types runs, an int
    literal being of every integer type that holds it. A call none fits, or two fit as well, is
    refused naming what the stub takes: `warp.sum(1)` is, as the literal fits four types.
    """

    def defined(stub: F) -> F:
        name = stub.__qualname__
        running = {
            kinds: implementing(name, kinds, returns, implementations)
            for kinds, returns in taken(stub).items()
        }
        if idle := set(implementations) - set(running.values()):
            raise TypeError(f"{name}: {idle.pop()!r} implements none of its signatures")
        return as_dispatching(stub, partial(_resolved, name, running))

    return defined


def typed_call(context, device: Callable, operands: Sequence[types.Type]) -> templates.Signature:
    """How Numba types a call of the device function or intrinsic `device` on `operands`."""
    return context.resolve_value_type(device).get_call_type(context, operands, {})


def lowered_call(
    device: Callable, context, builder: ir.IRBuilder, call: templates.Signature, values: Sequence
) -> ir.Value:
    """Call `device` as `call` typed it, linking whatever it was compiled with."""
    function = context.typing_context.resolve_value_type(device)
    compiled = context.get_function(function, call)
    context.add_linking_libs(getattr(compiled, "libs", ()))
    return compiled(builder, values)


def _resolved(
    name: str, running: Mapping[Kinds, Callable], context, operands: Sequence[types.Type]
) -> tuple[templates.Signature, Callable[..., ir.Value]]:
    """The call of the signature `operands` fit best, typed and lowered at its declared types."""
    typed, device = _chosen(name, running, operands)
    return typed_call(context, device, typed), partial(lowered_call, device)


def _chosen(
    name: str, running: Mapping[Kinds, Callable], operands: Sequence[types.Type]
) -> tuple[list[types.Type], Callable]:
    """The Numba types of the signature `operands` fit best, and what runs it.

    A call that no signature fits, or that two fit as well, is refused.
    """
    given = [types.unliteral(operand) for operand in operands]
    numba_kinds = {kinds: [numba_type(kind) for kind in kinds] for kinds in running}
    exact = {
        kinds: sum(map(_is_exactly, typed, operands, strict=True))
        for kinds, typed in numba_kinds.items()
        if all(map(_can_take, typed, given, strict=True))
    }
    best = [kinds for kinds, count in exact.items() if count == max(exact.values())]
    if len(best) != 1:
        taken_by = "; ".join(map(listed, running))
        tied = f"; {' and '.join(map(listed, best))} fit equally well" if best else ""
        spelled = ", ".join(map(_spelled, operands))
        raise TypingError(f"{name} takes {taken_by}, not ({spelled}){tied}")
    return numba_kinds[best[0]], running[best[0]]


def _can_take(declared: types.Type | None, given: types.Type) -> bool:
    """Whether a parameter `declared` takes `given`: an integer converted, any other as it is."""
    return declared == given or all(isinstance(kind, types.Integer) for kind in (declared, given))


def _is_exactly(declared: types.Type | None, operand: types.Type) -> bool:
    """Whether `operand` is of the type `declared`.

    An int literal is of every integer type that holds it, as it takes the type of the integer it
    meets in arithmetic.
    """
    if isinstance(operand, types.IntegerLiteral) and isinstance(declared, types.Integer):
        held = np.iinfo(declared.name)
        return held.min <= operand.literal_value <= held.max
    return declared == types.unliteral(operand)


def _spelled(kind: types.Type) -> str:
    """`kind` as patos spells it: a scalar by its short name, lanes by theirs, a literal as is."""
    if isinstance(kind, types.IntegerLiteral):
        return str(kind.literal_value)
    if isinstance(kind, types.Integer | types.Float):
        return named(np.dtype(kind.name).type)
    return str(kind)
