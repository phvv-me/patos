"""The C integer rules device arithmetic types by.

Numba types every integer operation in 64 bits: `u64 + 1` becomes a signed `int64`, which costs
every index a negative-index fixup, `u32 + 1` an `int64` as well, and a local assigned an `i32` on
one branch and a `u64` on the other unifies to a `float64` that no array accepts. The rules here
type the operands' meeting as C does instead. A literal takes the type of the integer it meets
when it fits there, and two integers of different signedness meet in the unsigned type when it is
at least as wide as the signed one, else in the signed one, an operand narrower than 32 bits
first promoting to `i32`. Two integers of one signedness keep Numba's typing, 64 bits wide. A
module constant is no literal to Numba but an `int64`, so it meets a `u64` in the `u64` and widens
an `i32`.

`meet` reads the rules over the kinds the checker infers, and the backend's typing templates
(`patos.cuda.backend.arithmetic`) read the same function over Numba's types, so the checker and
the compiler cannot disagree.
"""

import ast

import numpy as np

from .declarations import Declared, IntLiteral, Kind, is_integer


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
