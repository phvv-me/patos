"""One device function or kernel as written, read the way C reads declarations: what each
parameter, return and declared local is."""

import annotationlib
import ast
import inspect
import textwrap
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from types import FunctionType, NoneType
from typing import TypeIs, TypeVar

import numpy as np

from ..scalars import ArrayOf, Lanes, canonical
from .declarations import (
    Declared,
    Evaluated,
    NamedValue,
    Record,
    Returns,
    Signature,
    Subject,
    declared,
    is_scalar,
    named,
)
from .items import items, items_through
from .scopes import annotation_scope, body_scope, resolved


class AnnotationError(TypeError):
    """A device function or kernel whose annotations are incomplete or contradict each other, or
    whose casts repeat them."""


class Reading:
    """One device function or kernel as written, with every issue its annotations raise.

    kernel: whether `function` is a kernel, which returns None.
    owner: the record class `function` is a member of, which types its unannotated `self`.
    """

    def __init__(
        self, function: FunctionType, *, kernel: bool = False, owner: type | None = None
    ) -> None:
        self.function, self.kernel, self.owner = function, kernel, owner
        self.members = annotation_scope(function, owner)
        self.definition = self._parsed(function)
        self.namespace = body_scope(function, self.walk(), self.definition.args.args)
        # Device code is generic over type variables only, its own and its record's.
        self.type_params = tuple(
            parameter
            for parameter in (*function.__type_params__, *getattr(owner, "__type_params__", ()))
            if isinstance(parameter, TypeVar)
        )
        self.issues: list[tuple[int, str]] = []
        self.parameters = {
            argument.arg: self._parameter(argument) for argument in self.definition.args.args
        }
        self.returns = self._returns(kernel=kernel)
        self.declared = self._declarations()
        self.variables = self._variables()

    def callee(self, node: ast.expr) -> Signature | None:
        """The signature of the device function `node` calls, or of the named value it builds."""
        if not isinstance(node, ast.Call):
            return None
        if (built := self.constructed(node)) is not None:
            return Signature(parameters=built.kinds(), returns=built)
        # A dispatcher keeps the signature on the function it compiles, an intrinsic on itself.
        callee = resolved(node.func, self.namespace)
        return getattr(getattr(callee, "py_func", callee), "device_signature", None)

    def calls(self, node: ast.AST, *functions: Callable) -> TypeIs[ast.Call]:
        """Whether `node` calls one of `functions`, through any name it was imported under."""
        return isinstance(node, ast.Call) and resolved(node.func, self.namespace) in functions

    def cast(self, node: ast.expr) -> type[np.generic] | Lanes | None:
        """The scalar type or lanes `node` converts to when it is a cast `T(value)`."""
        if not (isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords):
            return None
        target = resolved(node.func, self.namespace)
        if is_scalar(target):
            return canonical(target)
        return target if isinstance(target, Lanes) else None

    def constructed(self, node: ast.Call) -> NamedValue | None:
        """The named value `node` builds when it calls the class that declares it."""
        kind = declared(resolved(node.func, self.namespace))
        return kind if isinstance(kind, NamedValue) else None

    def declared_scalar(self, target: ast.expr) -> type[np.generic] | Lanes | None:
        """The scalar type or lanes `target` is declared, which each assignment converts to."""
        if isinstance(target, ast.Name):
            kind = self.declared.get(target.id)
            return kind if is_convertible(kind) else None
        return None

    def fields(self, node: ast.Call, built: NamedValue) -> list[ast.expr]:
        """The expression each field of `built` takes from the call `node`, a default included.

        A default is a literal, or a scalar written as its type (`step: i32 = i32(1)`).

        Raises `TypeError` unless the arguments fit the fields one by one.
        """
        if any(isinstance(argument, ast.Starred) for argument in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            raise TypeError("takes its fields one by one, not unpacked")
        given = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        bound = inspect.signature(built.cls).bind(*node.args, **given)
        bound.apply_defaults()
        return [
            value
            if isinstance(value, ast.expr)
            else ast.Constant(value.item() if isinstance(value, np.generic) else value)
            for value in bound.args
        ]

    def issue(self, node: ast.expr | ast.stmt | ast.arg, message: str) -> None:
        self.issues.append((node.lineno, message))

    def item_call(self, node: ast.AST) -> ast.Call | None:
        """The `items` or `items_through` call that `node` loops over, None for any other node."""
        if isinstance(node, ast.For) and self.calls(node.iter, items, items_through):
            return node.iter
        return None

    def raise_issues(self) -> None:
        """Raise `AnnotationError` naming every issue found, line by line, if there is one."""
        if self.issues:
            path, name = self.function.__code__.co_filename, self.function.__qualname__
            lines = [
                f"{path}:{line}: {name}: {message}" for line, message in sorted(set(self.issues))
            ]
            raise AnnotationError("\n".join(lines))

    def walk(self) -> Iterator[ast.AST]:
        """Every node of the body, nested definitions left out."""
        pending: list[ast.AST] = list(self.definition.body)
        while pending:
            node = pending.pop()
            yield node
            pending.extend(
                child
                for child in ast.iter_child_nodes(node)
                if not isinstance(child, ast.FunctionDef | ast.Lambda)
            )

    @staticmethod
    def _parsed(function: FunctionType) -> ast.FunctionDef:
        """The definition of `function` as a tree numbered like its file."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
        definition = tree.body[0]
        if not isinstance(definition, ast.FunctionDef):
            raise AnnotationError(f"{function.__qualname__} is not a plain function definition")
        return definition

    def _annotation(self, node: ast.expr) -> Declared:
        """Return what an annotation declares, where a type parameter declares None."""
        value = self._evaluated(node)
        if isinstance(value, TypeVar):
            return None
        kind = declared(value)
        if kind is None:
            self.issue(node, f"`{ast.unparse(node)}` names no device type")
        return kind

    def _check_loop(self, node: ast.For, declarations: Mapping[str, Declared]) -> None:
        for target in ast.walk(node.target):
            if isinstance(target, ast.Name) and is_convertible(declarations.get(target.id)):
                self.issue(node, f"loop variable `{target.id}` is declared; the iterable types it")

    def _declarations(self) -> dict[str, Declared]:
        """The type of every scalar or tuple parameter and declared local, one each."""
        declarations: dict[str, Declared] = {
            name: kind for name, kind in self.parameters.items() if not isinstance(kind, ArrayOf)
        }
        for node in sorted(self.walk(), key=lambda node: getattr(node, "lineno", 0)):
            if isinstance(node, ast.AnnAssign):
                self._declare(node, declarations)
        for node in self.walk():
            if isinstance(node, ast.For):
                self._check_loop(node, declarations)
        return declarations

    def _declare(self, node: ast.AnnAssign, declarations: MutableMapping[str, Declared]) -> None:
        """Record the type a declared local gets, flagging a declaration that cannot stand."""
        if not isinstance(node.target, ast.Name):
            self.issue(node, f"`{ast.unparse(node.target)}` is no name to declare")
            return
        name, kind = node.target.id, self._annotation(node.annotation)
        if isinstance(kind, ArrayOf):
            self.issue(node, f"`{name}` declares an array, which only a parameter can")
        elif name in self.parameters:
            self.issue(node, f"parameter `{name}` is declared again; annotate the parameter")
        elif name in declarations:
            self.issue(
                node,
                f"`{name}` is declared {named(declarations[name])} already; "
                "one declaration types every assignment",
            )
        declarations.setdefault(name, kind)

    def _evaluated(self, node: ast.expr) -> Evaluated:
        """What an annotation evaluates to where the function is defined, None for a name unbound.

        Python keeps no annotation of a local, so every annotation is read from the source alike.
        """
        try:
            return annotationlib.ForwardRef(ast.unparse(node)).evaluate(
                globals=self.function.__globals__,
                locals=self.members,
                type_params=self.type_params,
            )
        except NameError:
            return None

    def _parameter(self, argument: ast.arg) -> Declared:
        if argument.annotation is None and self.owner is not None and argument.arg == "self":
            return declared(self.owner)
        if argument.annotation is None:
            self.issue(argument, f"parameter `{argument.arg}` has no annotation")
            return None
        kind = self._annotation(argument.annotation)
        if self.kernel and isinstance(kind, NamedValue):
            self.issue(argument, f"a launch passes no {named(kind)}; a record carries its fields")
        if self.kernel and isinstance(kind, Lanes):
            self.issue(argument, f"a launch passes no {named(kind)}; pass a u32 and convert it")
        return kind

    def _returns(self, *, kernel: bool) -> Returns:
        node = self.definition.returns
        if node is None:
            self.issue(self.definition, "the return has no annotation")
            return None
        if isinstance(node, ast.Constant) and node.value is None:
            return NoneType
        if kernel:
            self.issue(node, "a kernel returns None")
        kind = self._annotation(node)
        if _has_array(kind):
            self.issue(node, "a device function returns no array")
        return kind

    def _variables(self) -> tuple[tuple[int, TypeVar], ...]:
        """The type parameter each parameter and the return declare, by index."""
        nodes = [argument.annotation for argument in self.definition.args.args]
        nodes.append(self.definition.returns)
        evaluated = ((index, self._evaluated(node)) for index, node in enumerate(nodes) if node)
        return tuple((index, kind) for index, kind in evaluated if isinstance(kind, TypeVar))


def is_convertible(kind: Subject) -> TypeIs[type[np.number] | Lanes]:
    """Whether a value converts to `kind` by a cast: a scalar type or packed lanes."""
    return is_scalar(kind) or isinstance(kind, Lanes)


def _has_array(declared: Returns) -> bool:
    if isinstance(declared, Record):
        return any(_has_array(kind) for kind in declared.cls.declarations().values())
    if isinstance(declared, NamedValue):
        return any(_has_array(kind) for kind in declared.kinds())
    if isinstance(declared, tuple):
        return any(_has_array(element) for element in declared)
    return isinstance(declared, ArrayOf)
