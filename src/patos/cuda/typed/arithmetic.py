"""The C integer rules device arithmetic types by, written once for the checker and for Numba.

Numba types every integer operation in 64 bits: `u64 + 1` becomes a signed `int64`, which costs
every index a negative-index fixup, `u32 + 1` an `int64` as well, and a local assigned an `i32` on
one branch and a `u64` on the other unifies to a `float64` that no array accepts. The rules here
type the operands' meeting as C does instead. A literal takes the type of the integer it meets
when it fits there, and two integers of different signedness meet in the unsigned type when it is
at least as wide as the signed one, else in the signed one, an operand narrower than 32 bits
first promoting to `i32`. Two integers of one signedness keep Numba's typing, 64 bits wide. A
module constant is no literal to Numba but an `int64`, so it meets a `u64` in the `u64` and widens
an `i32`.

`meet` reads the rules over the kinds the checker infers; the typing templates registered below
read the same rules over Numba's types, so the checker and the compiler cannot disagree.
"""

import ast
import operator
from collections.abc import Callable

import numpy as np
from numba import types
from numba.core.typing import templates
from numba.cuda.cudadecl import registry

from .declarations import Declared, IntLiteral, Kind, is_integer

_ARITHMETIC = (
    operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod,
    operator.and_, operator.or_, operator.xor,
    operator.iadd, operator.isub, operator.imul, operator.ifloordiv, operator.imod,
    operator.iand, operator.ior, operator.ixor,
    min, max,
)  # fmt: skip
_SHIFTS = (operator.lshift, operator.rshift, operator.ilshift, operator.irshift)
_COMPARISONS = (operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne)


def meet(left: Declared, right: Declared) -> type[np.integer] | None:
    """Return the type two integer operands meet in under the rules above, or None.

    None leaves the pair to Numba's own typing.
    """
    left, right = _promoted(left), _promoted(right)
    if is_integer(left) and isinstance(right, IntLiteral):
        return left if _is_within(right.value, left) else None
    if is_integer(right) and isinstance(left, IntLiteral):
        return right if _is_within(left.value, right) else None
    if is_integer(left) and is_integer(right) and _is_signed(left) != _is_signed(right):
        unsigned, signed = (right, left) if _is_signed(left) else (left, right)
        return unsigned if np.dtype(unsigned).itemsize >= np.dtype(signed).itemsize else signed
    return None


def combined(left: Declared, right: Declared) -> Kind:
    """Return the type two operands of arithmetic, `min` or `max` meet in.

    It follows the rules above, else as Numba types two integers of one signedness, in that
    signedness at 64 bits.
    """
    met = meet(left, right)
    if (
        met is None
        and is_integer(left)
        and is_integer(right)
        and _is_signed(left) == _is_signed(right)
    ):
        return np.int64 if _is_signed(left) else np.uint64
    return met


def operated(op: ast.operator, left: Declared, right: Declared) -> Kind:
    """Return the type `left op right` takes, or None where the checker cannot tell."""
    if isinstance(op, ast.Div | ast.Pow | ast.MatMult):
        return None
    if isinstance(op, ast.LShift | ast.RShift):
        return _shifted(left, right)
    return combined(left, right)


def met(left: types.Type, right: types.Type) -> types.Integer | None:
    """The Numba type two operands meet in under the rules above, or None for Numba to type."""
    kind = meet(_kind_of(left), _kind_of(right))
    return None if kind is None else getattr(types, np.dtype(kind).name)


def _kind_of(value: types.Type) -> Kind:
    """What the checker calls a Numba type: an int literal's value, or an integer's NumPy type."""
    if isinstance(value, types.IntegerLiteral):
        return IntLiteral(value=value.literal_value)
    if isinstance(value, types.Integer):
        return np.dtype(f"{'i' if value.signed else 'u'}{value.bitwidth // 8}").type
    return None


def _is_within(value: int, kind: type[np.integer]) -> bool:
    return np.iinfo(kind).min <= value <= np.iinfo(kind).max


def _is_signed(kind: type[np.integer]) -> bool:
    return issubclass(kind, np.signedinteger)


def _promoted(kind: Declared) -> Declared:
    return np.int32 if is_integer(kind) and np.dtype(kind).itemsize < 4 else kind


def _shifted(left: Declared, right: Declared) -> Kind:
    """Return the type `left` shifted by `right` takes, or None where the checker cannot tell."""
    met = meet(left, right)
    if met is not None and np.dtype(met).itemsize == 8:
        return met
    # Numba shifts in 64 bits, keeping the signedness of the value shifted.
    if isinstance(left, IntLiteral):
        return np.int64 if is_integer(right) or isinstance(right, IntLiteral) else None
    if is_integer(left) and (is_integer(right) or isinstance(right, IntLiteral)):
        return np.int64 if _is_signed(left) else np.uint64
    return None


def _rule(
    op: Callable, result: Callable[[types.Integer], types.Type], *, widest: bool = False
) -> type[templates.AbstractTemplate]:
    """A typing template for `op` under the rules above.

    It asks about the operands' literals before their plain types, since a literal takes the type
    of the integer it meets.

    widest: apply only where the operands meet in a 64-bit type, which numba-cuda's shift
        lowering requires: it widens a narrower shifted value to 64 bits whatever the signature.
    """

    class Rule(templates.AbstractTemplate):
        key = op
        prefer_literal = True

        def generic(self, args, kws) -> templates.Signature | None:
            if kws or len(args) != 2:
                return None
            meeting = met(*args)
            if meeting is None or (widest and meeting.bitwidth != 64):
                return None
            return templates.signature(result(meeting), meeting, meeting)

    return Rule


for _op in _ARITHMETIC:
    registry.register_global(_op)(_rule(_op, lambda meeting: meeting))
for _op in _COMPARISONS:
    registry.register_global(_op)(_rule(_op, lambda _: types.boolean))
for _op in _SHIFTS:
    registry.register_global(_op)(_rule(_op, lambda meeting: meeting, widest=True))
