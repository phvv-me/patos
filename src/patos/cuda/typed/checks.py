"""The checks a function's annotations pass before it compiles: every `return` converts element by
element, and no cast repeats what an annotation or an operator already does."""

import ast
from collections.abc import Callable, Iterator
from functools import partial
from types import FunctionType, NoneType

from .arithmetic import combined, operated
from .declarations import Declared, Kind, Returns, is_scalar, named
from .inference import Inference
from .reading import Reading
from .scalars import ArrayOf


def read(function: FunctionType, *, kernel: bool, owner: type | None = None) -> Reading:
    """`function` read and checked.

    owner: the record class `function` is a member of.
    Raises `AnnotationError` naming every line whose annotations are missing, contradict each
    other, or are repeated by a cast.
    """
    reading = Reading(function, kernel=kernel, owner=owner)
    Checker(reading, Inference(reading)).check()
    return reading


class Checker:
    """The checks one function passes, reported with the issues its reading raised."""

    def __init__(self, reading: Reading, inference: Inference) -> None:
        self.reading = reading
        self.inference = inference

    def check(self) -> None:
        """Raise `AnnotationError` naming every line whose annotations fail.

        An annotation fails when it is missing, contradicts another, or a cast repeats it.
        """
        for node in self.reading.walk():
            if isinstance(node, ast.Return):
                self._check_return(node)
            if isinstance(node, ast.Call):
                self._check_arguments(node)
            self._check_conversion(node)
        self.reading.raise_issues()

    def _check_arguments(self, call: ast.Call) -> None:
        """Flag every argument of a device function that casts to its parameter's own type."""
        callee = self.reading.callee(call)
        if callee is None:
            return
        for argument, parameter in zip(call.args, callee.parameters, strict=False):
            self._converted_by(
                parameter if is_scalar(parameter) else None,
                f"`{ast.unparse(call.func)}`'s parameter",
                argument,
            )

    def _check_conversion(self, node: ast.AST) -> None:
        """Flag `node` when it is a cast, or holds one, that something else already converts."""
        returns = self.reading.returns
        match node:
            case ast.Call() if (
                cast := self.reading.cast(node)
            ) is not None and self.inference.kind(node.args[0]) == cast:
                self._redundant(node, f"`{ast.unparse(node.args[0])}` is already {named(cast)}")
            case (
                ast.Assign(targets=[target], value=value)
                | ast.AnnAssign(target=target, value=value)
            ) if value is not None:
                self._converted_by(*self._stored(target), value)
            case ast.Return(value=ast.Tuple(elts=elements)) if isinstance(returns, tuple):
                for kind, element in zip(returns, elements, strict=False):
                    self._converted_by(kind, "the return annotation", element)
            case ast.Return(value=value) if value is not None and is_scalar(returns):
                self._converted_by(returns, "the return annotation", value)
            case ast.BinOp(left=left, op=op, right=right):
                self._operand(left, right, partial(operated, op))
                self._operand(right, left, partial(operated, op))
            case ast.AugAssign(target=target, op=op, value=value):
                self._operand(value, target, partial(operated, op))
            case (
                ast.Compare(left=left, comparators=[right])
                | ast.Call(func=ast.Name(id="min" | "max"), args=[left, right])
            ):
                self._operand(left, right, combined)
                self._operand(right, left, combined)

    def _check_return(self, node: ast.Return) -> None:
        returns, value = self.reading.returns, node.value
        if returns is NoneType:
            if value is not None:
                self.reading.issue(node, "returns a value where None is declared")
        elif value is None:
            self.reading.issue(node, f"returns nothing where {named(returns)} is declared")
        elif isinstance(returns, tuple) and not self._returns_elements(value, returns):
            self.reading.issue(node, f"return the {len(returns)} elements so each converts")

    def _converted_by(self, kind: Returns, converter: str, value: ast.expr) -> None:
        for branch in _branches(value):
            if kind is not None and self.reading.cast(branch) is kind:
                self._redundant(branch, f"{converter} converts to {named(kind)}")

    def _operand(
        self, operand: ast.expr, other: ast.expr, meet: Callable[[Declared, Declared], Kind]
    ) -> None:
        """Flag `operand` when it casts to the type `other` has, which the operator converts to."""
        cast = self.reading.cast(operand)
        if (
            cast is None
            or not isinstance(operand, ast.Call)
            or self.inference.kind(other) is not cast
        ):
            return
        if meet(cast, self.inference.kind(operand.args[0])) is cast:
            self._redundant(
                operand, f"`{ast.unparse(other)}` is {named(cast)}, so the operator converts it"
            )

    def _redundant(self, cast: ast.expr, reason: str) -> None:
        self.reading.issue(cast, f"`{ast.unparse(cast)}` is a redundant cast: {reason}")

    def _returns_elements(self, value: ast.expr, returns: tuple[Declared, ...]) -> bool:
        """Whether `value` is a tuple the return annotation converts element by element.

        The call of a device function returning the same types is one too.
        """
        if isinstance(value, ast.Tuple):
            return len(value.elts) == len(returns)
        if isinstance(value, ast.Name):
            return self.inference.kinds.get(value.id) == returns
        callee = self.reading.callee(value)
        return callee is not None and callee.returns == returns

    def _stored(self, target: ast.expr) -> tuple[Kind, str]:
        """The type a store to `target` converts to, and what does: a declaration or an array."""
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            array = self.reading.parameters.get(target.value.id)
            if isinstance(array, ArrayOf) and array.concrete:
                return array.element, f"the store into `{target.value.id}`"
        return self.reading.declared_scalar(target), f"the declaration of `{ast.unparse(target)}`"


def _branches(value: ast.expr) -> Iterator[ast.expr]:
    """The expressions a conversion of `value` lands on, through both branches of a conditional."""
    if isinstance(value, ast.IfExp):
        yield from _branches(value.body)
        yield from _branches(value.orelse)
    else:
        yield value
