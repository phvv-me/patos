"""One device function name over several typed implementations, as C++ overloads a name, and the
calls of a device function that a choice forwards to."""

import inspect
import operator
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import FunctionType

import numpy as np
from llvmlite import ir
from numba import types
from numba.core.errors import TypingError
from numba.core.typing import templates

from .declarations import Declared, Signature, named, numba_type
from .intrinsics import as_dispatching
from .lanes import LaneType


def dispatched[F: FunctionType](*implementations: Callable) -> Callable[[F], F]:
    """Make a stub one device function over typed `implementations`, as C++ overloads a name.

    A call runs the implementation its operands fit best, picked at compile time. One fits when
    each lanes operand is the lanes its parameter declares and each integer one converts to its
    parameter; of those, the one declaring the most operands' own types runs. A call none fits,
    or two fit as well, is refused naming what the stub takes.
    """

    def defined(stub: F) -> F:
        declared = {found: _parameters(found) for found in implementations}
        arity = len(inspect.signature(stub).parameters)
        if any(len(kinds) != arity for kinds in declared.values()):
            raise TypeError(f"{stub.__qualname__}: an implementation takes other operands")
        return as_dispatching(stub, partial(_resolved, stub.__qualname__, declared))

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
    name: str,
    candidates: Mapping[Callable, tuple[Declared, ...]],
    context,
    operands: Sequence[types.Type],
) -> tuple[templates.Signature, Callable[..., ir.Value]]:
    """The call of the implementation `operands` fit best, typed and lowered as it is."""
    device = _chosen(name, candidates, operands)
    return typed_call(context, device, operands), partial(lowered_call, device)


def _parameters(implementation: Callable) -> tuple[Declared, ...]:
    """What a device function or a PTX stub declares its parameters to be."""
    compiled = getattr(implementation, "py_func", implementation)
    signature: Signature | None = getattr(compiled, "device_signature", None)
    if signature is None:
        raise TypeError(f"{implementation!r} is no device function patos typed")
    return signature.parameters


def _chosen(
    name: str,
    candidates: Mapping[Callable, tuple[Declared, ...]],
    operands: Sequence[types.Type],
) -> Callable:
    """The implementation `operands` fit best, refused unless it is the only one."""
    given = [types.unliteral(operand) for operand in operands]
    numba_kinds = {
        found: [numba_type(kind) for kind in kinds] for found, kinds in candidates.items()
    }
    exact = {
        found: sum(map(operator.eq, declared, given, strict=True))
        for found, declared in numba_kinds.items()
        if all(map(_can_take, declared, given, strict=True))
    }
    best = [found for found, count in exact.items() if count == max(exact.values())]
    if len(best) != 1:
        taken = "; ".join(f"({', '.join(map(named, kinds))})" for kinds in candidates.values())
        raise TypingError(f"{name} takes {taken}, not ({', '.join(map(_spelled, given))})")
    return best[0]


def _can_take(declared: types.Type | None, given: types.Type) -> bool:
    """Whether a parameter `declared` takes `given`: lanes as they are, an integer converted."""
    if isinstance(declared, LaneType) or isinstance(given, LaneType):
        return declared == given
    return isinstance(declared, types.Integer) and isinstance(given, types.Integer)


def _spelled(kind: types.Type) -> str:
    """`kind` as patos spells it: a scalar by its short name, lanes by theirs."""
    if isinstance(kind, types.Integer):
        return named(np.dtype(kind.name).type)
    return str(kind)
