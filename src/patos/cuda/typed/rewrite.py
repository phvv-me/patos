"""The rewrite that makes annotations convert what they declare, and the function it rebuilds.

Every `return` converts to the return annotation and every assignment to a declared local to its
declaration, a conditional expression converting branch by branch so mixed branches never unify
through a float. Parameters need nothing: a device function compiles at the types its parameters
declare (`decorators`) and a kernel's launch converts its scalars (`kernels`). The annotations
are then dropped, and Numba compiles the IR explicit casts would have given.
"""

import ast
from collections.abc import Sequence
from types import CodeType, FunctionType

import numpy as np

from .declarations import Returns, Signature, is_scalar
from .reading import Reading


class Rewrite:
    """One function read by `Reading`, rebuilt so its annotations convert what they declare."""

    def __init__(self, function: Reading) -> None:
        self.function = function

    def rebuilt(self) -> FunctionType:
        """The function with every conversion in place and its annotations dropped.

        It carries the signature its callers' checks read, as `device_signature`.
        """
        original = self.function.function
        # The module's own globals, updated in place, so a name the module defines later resolves.
        namespace = self.function.namespace
        namespace.update(_patos_numpy=np)
        rebuilt = FunctionType(self._code(), namespace, original.__name__)
        rebuilt.__doc__ = original.__doc__
        rebuilt.__qualname__ = original.__qualname__
        rebuilt.__dict__["device_signature"] = Signature(
            tuple(self.function.parameters.values()), self.function.returns
        )
        return rebuilt

    def _code(self) -> CodeType:
        """The code object of the rewritten definition, compiled where the original was."""
        definition = _Conversions(self.function).visit(self.function.definition)
        definition.decorator_list = []
        definition.type_params = []
        definition.returns = None
        for argument in definition.args.args:
            argument.annotation = None
        ast.fix_missing_locations(self.function.tree)
        module = compile(self.function.tree, self.function.function.__code__.co_filename, "exec")
        return next(
            constant
            for constant in module.co_consts
            if getattr(constant, "co_name", None) == definition.name
        )


class _Conversions(ast.NodeTransformer):
    """Rewrite one function so every return and every assignment to a declared local converts."""

    def __init__(self, function: Reading) -> None:
        self.function = function

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
                self.function.declared_scalar(target) for target in targets
            ):
                return self._unpacked(targets, value, node)
        node.value = value
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        value = self.visit(node.value)
        target = node.target
        if not isinstance(target, ast.Name) or self.function.declared_scalar(target) is None:
            node.value = value
            return node
        current = ast.Name(id=target.id, ctx=ast.Load())
        stored = ast.Name(id=target.id, ctx=ast.Store())
        return self._assigned(stored, ast.BinOp(left=current, op=node.op, right=value), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self.generic_visit(node) if node is self.function.definition else node

    def visit_Return(self, node: ast.Return) -> ast.Return:
        if node.value is None:
            return node
        value = self._returned(self.function.returns, self.visit(node.value))
        return ast.copy_location(ast.Return(value=value), node)

    def _assigned(self, target: ast.expr, value: ast.expr, node: ast.stmt) -> ast.Assign:
        kind = self.function.declared_scalar(target)
        if kind is not None:
            value = _converted(kind, value)
        return ast.copy_location(ast.Assign(targets=[target], value=value), node)

    def _returned(self, declared: Returns, value: ast.expr) -> ast.expr:
        """`value` converted to what the return declares, through nested tuple displays."""
        if is_scalar(declared):
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
            if isinstance(target, ast.Name) and self.function.declared_scalar(target):
                received = ast.Name(id=f"_received_{target.id}", ctx=ast.Store())
                loaded = ast.Name(id=received.id, ctx=ast.Load())
                conversions.append(self._assigned(target, loaded, node))
                unpacked.append(received)
            else:
                unpacked.append(target)
        assign = ast.Assign(targets=[ast.Tuple(elts=unpacked, ctx=ast.Store())], value=value)
        return [ast.copy_location(assign, node), *conversions]


def _converted(kind: type[np.generic], value: ast.expr) -> ast.expr:
    """`kind(value)`, pushed into both branches of a conditional expression."""
    if isinstance(value, ast.IfExp):
        return ast.IfExp(
            test=value.test,
            body=_converted(kind, value.body),
            orelse=_converted(kind, value.orelse),
        )
    return ast.Call(func=_numpy(kind), args=[value])


def _numpy(kind: type[np.generic]) -> ast.Attribute:
    """The expression naming scalar type `kind` in a rewritten function."""
    return ast.Attribute(
        value=ast.Name(id="_patos_numpy", ctx=ast.Load()), attr=kind.__name__, ctx=ast.Load()
    )
