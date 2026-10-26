"""The C integer rules of `patos.cuda.front.arithmetic` as Numba typing templates, in force only
while patos compiles.

An operator's typing has no home but numba-cuda's shared registry, so the templates sit there, put
there by the first patos compile rather than by an import. They answer only inside `rules()`, the
scope every patos compile runs in: any other kernel the process compiles, such as one cuda.compute
builds, keeps Numba's own typing.
"""

import operator
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache

import numpy as np
from numba import types
from numba.core.typing import templates
from numba.cuda.cudadecl import registry

from ..front.arithmetic import meet
from ..front.declarations import IntLiteral, Kind

_ARITHMETIC = (
    operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod,
    operator.and_, operator.or_, operator.xor,
    operator.iadd, operator.isub, operator.imul, operator.ifloordiv, operator.imod,
    operator.iand, operator.ior, operator.ixor,
    min, max,
)  # fmt: skip
_SHIFTS = (operator.lshift, operator.rshift, operator.ilshift, operator.irshift)
_COMPARISONS = (operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne)
# Whether the code being typed is patos's, set by `rules`.
_COMPILING: ContextVar[bool] = ContextVar("patos_compiling", default=False)


@contextmanager
def rules() -> Generator[None]:
    """Type the integer arithmetic compiled inside the block by the C rules."""
    _registered()
    token = _COMPILING.set(True)
    try:
        yield
    finally:
        _COMPILING.reset(token)


def met(left: types.Type, right: types.Type) -> types.Integer | None:
    """The Numba type two operands meet in under the C rules, or None for Numba to type."""
    kind = meet(_kind_of(left), _kind_of(right))
    return None if kind is None else getattr(types, np.dtype(kind).name)


def _kind_of(value: types.Type) -> Kind:
    """What the checker calls a Numba type: an int literal's value, or an integer's NumPy type."""
    if isinstance(value, types.IntegerLiteral):
        return IntLiteral(value=value.literal_value)
    if isinstance(value, types.Integer):
        return np.dtype(f"{'i' if value.signed else 'u'}{value.bitwidth // 8}").type
    return None


def _rule(
    op: Callable, result: Callable[[types.Integer], types.Type], *, widest: bool = False
) -> type[templates.AbstractTemplate]:
    """A typing template for `op` under the C rules, answering inside `rules()` only.

    It asks about the operands' literals before their plain types, since a literal takes the type
    of the integer it meets.

    widest: apply only where the operands meet in a 64-bit type, which numba-cuda's shift
        lowering requires: it widens a narrower shifted value to 64 bits whatever the signature.
    """

    class Rule(templates.AbstractTemplate):
        key = op
        prefer_literal = True

        def generic(self, args, kws) -> templates.Signature | None:
            if kws or len(args) != 2 or not _COMPILING.get():
                return None
            meeting = met(*args)
            if meeting is None or (widest and meeting.bitwidth != 64):
                return None
            return templates.signature(result(meeting), meeting, meeting)

    return Rule


@cache
def _registered() -> None:
    """Put the templates on numba-cuda's typing registry, once."""
    for op in _ARITHMETIC:
        registry.register_global(op)(_rule(op, lambda meeting: meeting))
    for op in _COMPARISONS:
        registry.register_global(op)(_rule(op, lambda _: types.boolean))
    for op in _SHIFTS:
        registry.register_global(op)(_rule(op, lambda meeting: meeting, widest=True))
