"""The type every local and expression of one function takes, as far as its annotations tell."""

import ast
import operator
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from functools import partial

import numpy as np

from .arithmetic import combined, operated
from .declarations import (
    Declared,
    Evaluated,
    IntLiteral,
    Kind,
    Record,
    is_integer,
    is_scalar,
)
from .reading import Reading, resolved
from .scalars import ArrayOf

# One way a local gets its value: a reading of what is assigned, given the other locals.
type _Origin = Callable[[dict[str, Declared]], Declared]
# The operators Python folds when both operands are literals, so Numba sees one literal.
_FOLDED: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}  # fmt: skip


class Inference:
    """The type every local of one function takes, and the type of any of its expressions."""

    def __init__(self, function: Reading) -> None:
        self.function = function
        self.kinds = self._inferred()

    def kind(self, node: ast.expr, kinds: dict[str, Declared] | None = None) -> Declared:
        """The type `node` takes in the compiled function, or None where this cannot be told."""
        kinds = self.kinds if kinds is None else kinds
        match node:
            case ast.Constant(value=bool()):
                return bool
            case ast.Constant(value=int() as value):
                return IntLiteral(value)
            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=int() as value)) if (
                not isinstance(value, bool)
            ):
                return IntLiteral(-value)
            case ast.Name(id=name) if name in kinds:
                return kinds[name]
            case ast.Attribute(value=ast.Name(id=name), attr=field) if isinstance(
                record := kinds.get(name), Record
            ):
                kind = record.field(field)
                return kind if kind is bool or is_scalar(kind) else None
            case ast.Subscript(
                value=ast.Attribute(value=ast.Name(id=name), attr=field), slice=index
            ) if isinstance(record := kinds.get(name), Record) and not isinstance(
                index, ast.Slice
            ):
                array = record.field(field)
                return array.element if isinstance(array, ArrayOf) and array.concrete else None
            case ast.Name() | ast.Attribute():
                return self._constant(resolved(node, self.function.namespace))
            case ast.Subscript(
                value=ast.Name(id=name), slice=ast.Constant(value=int() as index)
            ) if isinstance(elements := kinds.get(name), tuple):
                return _scalar_at(elements, index)
            case ast.Subscript(value=ast.Name(id=name), slice=index) if not isinstance(
                index, ast.Slice
            ):
                array = self.function.parameters.get(name)
                return array.element if isinstance(array, ArrayOf) and array.concrete else None
            case ast.Call(func=ast.Name(id="min" | "max"), args=[left, right]):
                return combined(self.kind(left, kinds), self.kind(right, kinds))
            case ast.Call():
                if (cast := self.function.cast(node)) is not None:
                    return cast
                callee = self.function.callee(node)
                returns = callee.returns if callee is not None else None
                if returns is bool:
                    return bool
                return returns if is_scalar(returns) else None
            case ast.BinOp(left=left, op=op, right=right):
                left, right = self.kind(left, kinds), self.kind(right, kinds)
                if (
                    isinstance(left, IntLiteral)
                    and isinstance(right, IntLiteral)
                    and type(op) in _FOLDED
                ):
                    return IntLiteral(_FOLDED[type(op)](left.value, right.value))
                return operated(op, left, right)
            case ast.Compare() | ast.UnaryOp(op=ast.Not()):
                return bool
            case ast.UnaryOp(op=ast.Invert(), operand=operand):
                kind = self.kind(operand, kinds)
                return kind if is_integer(kind) and np.dtype(kind).itemsize == 8 else None
            case ast.IfExp(body=body, orelse=orelse):
                kind = self.kind(body, kinds)
                return (
                    kind
                    if kind == self.kind(orelse, kinds) and not isinstance(kind, IntLiteral)
                    else None
                )
        return None

    @staticmethod
    def _agreed(name: str, found: Sequence[_Origin], kinds: dict[str, Declared]) -> Kind:
        """The one scalar type every source of `name` gives, supposing `name` already has it."""
        readings = (source(kinds) for source in found)
        first = next((kind for kind in readings if is_scalar(kind)), None)
        supposed = kinds | {name: first}
        return (
            first
            if first is not None and all(source(supposed) == first for source in found)
            else None
        )

    @staticmethod
    def _constant(value: Evaluated) -> Kind:
        """What Numba types a module constant as: an int is `int64`, or `uint64` past its range."""
        if isinstance(value, bool):
            return bool
        if isinstance(value, int):
            return np.uint64 if value > np.iinfo(np.int64).max else np.int64
        kind = type(value)
        return kind if is_scalar(kind) else None

    def _augmented(
        self, name: str, op: ast.operator, value: ast.expr, kinds: dict[str, Declared]
    ) -> Kind:
        return operated(op, kinds.get(name), self.kind(value, kinds))

    def _bindings(self, node: ast.AST) -> Iterator[tuple[str, _Origin]]:
        """The local names `node` binds, each with the reading of the value it gets."""
        match node:
            case ast.Assign(targets=[ast.Name(id=name)], value=value):
                yield name, partial(self.kind, value)
            case ast.Assign(targets=[ast.Tuple(elts=targets)], value=value):
                yield from (
                    (target.id, partial(self._element, value, index))
                    for index, target in enumerate(targets)
                    if isinstance(target, ast.Name)
                )
            case ast.AugAssign(target=ast.Name(id=name), op=op, value=value):
                yield name, partial(self._augmented, name, op, value)
            case (
                ast.For(target=target)
                | ast.comprehension(target=target)
                | ast.NamedExpr(target=target)
            ):
                yield from (
                    (name.id, lambda _: None)
                    for name in ast.walk(target)
                    if isinstance(name, ast.Name)
                )

    def _element(self, value: ast.expr, index: int, kinds: dict[str, Declared]) -> Declared:
        """The type of element `index` of a tuple `value` unpacks into."""
        if isinstance(value, ast.Tuple) and index < len(value.elts):
            return self.kind(value.elts[index], kinds)
        if isinstance(value, ast.Name) and isinstance(elements := kinds.get(value.id), tuple):
            return _scalar_at(elements, index)
        callee = self.function.callee(value)
        returns = callee.returns if callee is not None else None
        return returns[index] if isinstance(returns, tuple) and index < len(returns) else None

    def _inferred(self) -> dict[str, Declared]:
        """Return the type of every local: its declaration, else what its assignments agree on.

        Agreement is as far as this reading can tell, so a cast of the local can be judged too.
        """
        sources: dict[str, list[_Origin]] = defaultdict(list)
        for node in self.function.walk():
            for name, source in self._bindings(node):
                sources[name].append(source)
        undeclared = {
            name: found for name, found in sources.items() if name not in self.function.declared
        }
        stored = [
            node.id
            for node in self.function.walk()
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        ]
        kinds: dict[str, Declared] = (
            dict.fromkeys([*self.function.parameters, *stored]) | self.function.declared
        )
        for _ in range(len(undeclared) + 1):
            agreed = {name: self._agreed(name, found, kinds) for name, found in undeclared.items()}
            if all(kinds[name] == kind for name, kind in agreed.items()):
                break
            kinds |= agreed
        return kinds


def _scalar_at(elements: Sequence[Declared], index: int) -> Kind:
    """Return element `index` of a tuple reading when it is a scalar type, else None."""
    element = elements[index] if index < len(elements) else None
    return element if is_scalar(element) else None
