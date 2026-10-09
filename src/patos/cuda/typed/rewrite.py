"""The rewrite that makes annotations convert what they declare, and the function it rebuilds.

Every `return` converts to the return annotation and every assignment to a declared local to its
declaration, a conditional expression converting branch by branch so mixed branches never unify
through a float. Parameters need nothing: a device function compiles at the types its parameters
declare (`decorators`) and a kernel's launch converts its scalars (`kernels`). The annotations
are then dropped, and Numba compiles the IR explicit casts would have given.

A kernel's `for item in items(count)` becomes the `while` loop a kernel writes by hand over its
own items, so the PTX is the same. A named value built by calling its class converts each scalar
field to its declaration.
"""

import ast
import copy
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from types import CodeType, FunctionType
from typing import Protocol

import numpy as np

from .. import scalars
from ..scalars import Lanes, i32, i64
from .declarations import Returns, Signature
from .items import items_through
from .reading import Reading, is_convertible


class Helper(Protocol):
    """A device function of no parameters, such as the helpers of `identity`."""

    __name__: str

    def __call__(self) -> i32 | i64: ...


@dataclass(frozen=True)
class Items:
    """Where a kernel's items start and how far apart they are, device functions of its `per`.

    first: gives the thread's first item.
    stride: gives the step to its next, None for a kernel that does not stride.
    """

    first: Helper
    stride: Helper | None

    def bindings(self) -> dict[str, Helper]:
        """The device functions by the reserved names a rewritten loop calls them by."""
        return {
            _bound(function): function
            for function in (self.first, self.stride)
            if function is not None
        }


def _bound(function: Helper) -> str:
    return f"_patos_{function.__name__}"


class Rewrite:
    """One function read by `Reading`, rebuilt so its annotations convert what they declare."""

    def __init__(self, reading: Reading, items: Items | None = None) -> None:
        self.reading = reading
        self.items = items

    def rebuilt(self) -> FunctionType:
        """The function with every conversion in place and its annotations dropped.

        It carries the signature its callers' checks read, as `device_signature`.
        """
        original = self.reading.function
        # The module's own globals, updated in place, so a name the module defines later resolves.
        namespace = self.reading.namespace
        namespace.update(_patos_numpy=np, _patos_scalars=scalars)
        if self.items is not None:
            namespace.update(self.items.bindings())
        rebuilt = FunctionType(self._code(), namespace, original.__name__)
        rebuilt.__doc__ = original.__doc__
        rebuilt.__qualname__ = original.__qualname__
        rebuilt.__dict__["device_signature"] = Signature(
            tuple(self.reading.parameters.values()), self.reading.returns
        )
        return rebuilt

    def _code(self) -> CodeType:
        """The code object of the rewritten definition, compiled where the original was."""
        definition = _Conversions(self.reading).visit(self.reading.definition)
        if self.items is not None:
            definition = _ItemLoops(self.reading, self.items).visit(definition)
            self.reading.raise_issues()
        definition.decorator_list = []
        definition.type_params = []
        definition.returns = None
        for argument in definition.args.args:
            argument.annotation = None
        ast.fix_missing_locations(self.reading.tree)
        module = compile(self.reading.tree, self.reading.function.__code__.co_filename, "exec")
        return next(
            constant
            for constant in module.co_consts
            if getattr(constant, "co_name", None) == definition.name
        )


class _Conversions(ast.NodeTransformer):
    """Rewrite one function so every return and every assignment to a declared local converts."""

    def __init__(self, reading: Reading) -> None:
        self.reading = reading

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt | None:
        if node.value is None:
            return None
        return self._assigned(node.target, self.visit(node.value), node)

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | list[ast.stmt]:
        value = self.visit(node.value)
        match node.targets:
            case [ast.Name() as target]:
                return self._assigned(target, value, node)
            case [ast.Tuple(elts=targets)] if any(
                self.reading.declared_scalar(target) for target in targets
            ):
                return self._unpacked(targets, value, node)
        node.value = value
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        value = self.visit(node.value)
        target = node.target
        if not isinstance(target, ast.Name) or self.reading.declared_scalar(target) is None:
            node.value = value
            return node
        current = ast.Name(id=target.id, ctx=ast.Load())
        stored = ast.Name(id=target.id, ctx=ast.Store())
        return self._assigned(stored, ast.BinOp(left=current, op=node.op, right=value), node)

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if (built := self.reading.constructed(node)) is not None:
            fields = self.reading.fields(node, built)
            node.args = [
                _converted(kind, value) if is_convertible(kind) else value
                for kind, value in zip(built.kinds(), fields, strict=True)
            ]
            node.keywords = []
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self.generic_visit(node) if node is self.reading.definition else node

    def visit_Return(self, node: ast.Return) -> ast.Return:
        if node.value is None:
            return node
        value = self._returned(self.reading.returns, self.visit(node.value))
        return ast.copy_location(ast.Return(value=value), node)

    def _assigned(self, target: ast.expr, value: ast.expr, node: ast.stmt) -> ast.Assign:
        kind = self.reading.declared_scalar(target)
        if kind is not None:
            value = _converted(kind, value)
        return ast.copy_location(ast.Assign(targets=[target], value=value), node)

    def _returned(self, declared: Returns, value: ast.expr) -> ast.expr:
        """`value` converted to what the return declares, through nested tuple displays."""
        if is_convertible(declared):
            return _converted(declared, value)
        if isinstance(declared, tuple) and isinstance(value, ast.Tuple):
            value.elts = [
                self._returned(kind, element)
                for kind, element in zip(declared, value.elts, strict=True)
            ]
        return value

    def _unpacked(
        self, targets: Sequence[ast.expr], value: ast.expr, node: ast.Assign
    ) -> list[ast.stmt]:
        """Unpack into temporaries where a target is declared, then convert them in order."""
        unpacked: list[ast.expr] = []
        conversions: list[ast.stmt] = []
        for target in targets:
            if isinstance(target, ast.Name) and self.reading.declared_scalar(target):
                received = ast.Name(id=f"_received_{target.id}", ctx=ast.Store())
                loaded = ast.Name(id=received.id, ctx=ast.Load())
                conversions.append(self._assigned(target, loaded, node))
                unpacked.append(received)
            else:
                unpacked.append(target)
        assign = ast.Assign(targets=[ast.Tuple(elts=unpacked, ctx=ast.Store())], value=value)
        return [ast.copy_location(assign, node), *conversions]


class _ItemLoops(ast.NodeTransformer):
    """Rewrite every `for item in items(count)` into the loop a kernel writes by hand.

    A kernel that strides gets a `while` loop over a hidden cursor, which is copied to `item` at
    the top of each pass so that the body may assign `item`; the cursor advances at the bottom and
    before every `continue`. Any other kernel has one item, so the loop is the guard `if item <
    count`, and a `break` or `continue` has nothing to leave.
    """

    def __init__(self, reading: Reading, items: Items) -> None:
        self.reading = reading
        self.items = items
        self.loops = itertools.count()

    def visit_For(self, node: ast.For) -> ast.stmt | list[ast.stmt]:
        self.generic_visit(node)
        call, item = self.reading.item_call(node), node.target
        if call is None or not isinstance(item, ast.Name):
            return node
        compare = ast.LtE() if self.reading.calls(call, items_through) else ast.Lt()
        if self.items.stride is None:
            return self._guarded(node, item, call, compare)
        return self._strided(node, call, compare, self.items.stride)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self.generic_visit(node) if node is self.reading.definition else node

    def _guarded(
        self, node: ast.For, item: ast.Name, call: ast.Call, compare: ast.cmpop
    ) -> list[ast.stmt]:
        start = ast.Assign(targets=[item], value=_call(self.items.first))
        test = ast.Compare(left=_name(item.id), ops=[compare], comparators=call.args)
        body = _Exits(self.reading, None).statements(node.body)
        return [start, ast.copy_location(ast.If(test=test, body=body, orelse=[]), node)]

    def _strided(
        self, node: ast.For, call: ast.Call, compare: ast.cmpop, stride: Helper
    ) -> list[ast.stmt]:
        number = next(self.loops)
        cursor, step = f"_patos_item{number}", f"_patos_stride{number}"
        advance = ast.AugAssign(target=_name(cursor, ast.Store()), op=ast.Add(), value=_name(step))
        test = ast.Compare(left=_name(cursor), ops=[compare], comparators=call.args)
        item = ast.Assign(targets=[node.target], value=_name(cursor))
        body = [item, *_Exits(self.reading, advance).statements(node.body), advance]
        return [
            ast.Assign(targets=[_name(cursor, ast.Store())], value=_call(self.items.first)),
            ast.Assign(targets=[_name(step, ast.Store())], value=_call(stride)),
            ast.copy_location(ast.While(test=test, body=body, orelse=[]), node),
        ]


class _Exits(ast.NodeTransformer):
    """The statements of one loop's body, advancing its cursor before every `continue`.

    A loop with no cursor to advance has no pass to leave early, so it flags both exits.
    """

    def __init__(self, reading: Reading, advance: ast.stmt | None) -> None:
        self.reading = reading
        self.advance = advance

    def statements(self, body: Sequence[ast.stmt]) -> list[ast.stmt]:
        module = ast.Module(body=list(body), type_ignores=[])
        self.generic_visit(module)
        return module.body

    def visit_Break(self, node: ast.Break) -> ast.stmt:
        return node if self.advance is not None else self._refused(node)

    def visit_Continue(self, node: ast.Continue) -> list[ast.stmt] | ast.stmt:
        if self.advance is None:
            return self._refused(node)
        return [copy.deepcopy(self.advance), node]

    def visit_For(self, node: ast.For | ast.While) -> ast.stmt:
        node.orelse = self.statements(node.orelse)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.stmt:
        return node

    visit_While = visit_For

    def _refused(self, node: ast.stmt) -> ast.stmt:
        self.reading.issue(node, "a kernel that does not stride has one item: `return` leaves it")
        return node


def _call(function: Helper) -> ast.Call:
    return ast.Call(func=_name(_bound(function)), args=[], keywords=[])


def _name(identifier: str, context: ast.expr_context | None = None) -> ast.Name:
    return ast.Name(id=identifier, ctx=context or ast.Load())


def _converted(kind: type[np.generic] | Lanes, value: ast.expr) -> ast.expr:
    """`kind(value)`, pushed into both branches of a conditional expression."""
    if isinstance(value, ast.IfExp):
        return ast.IfExp(
            test=value.test,
            body=_converted(kind, value.body),
            orelse=_converted(kind, value.orelse),
        )
    return ast.Call(func=_converter(kind), args=[value])


def _converter(kind: type[np.generic] | Lanes) -> ast.Attribute:
    """The expression naming scalar type or lanes `kind` in a rewritten function."""
    module, name = (
        ("_patos_scalars", kind.name)
        if isinstance(kind, Lanes)
        else ("_patos_numpy", kind.__name__)
    )
    return ast.Attribute(value=_name(module), attr=name, ctx=ast.Load())
