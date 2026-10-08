"""The rewrite that makes annotations convert what they declare, and the function it rebuilds.

Every `return` converts to the return annotation and every assignment to a declared local to its
declaration, a conditional expression converting branch by branch so mixed branches never unify
through a float. Every `Array[T]` parameter is checked on entry, and a scalar parameter converts
there unless Numba's own signature declares it, as it does a device function's (`decorators`).
The annotations are then dropped, and Numba compiles the IR explicit casts would have given.
"""

import ast
import itertools
from collections.abc import Sequence
from types import CodeType, FunctionType, ModuleType

import numpy as np
from numba import types
from numba.core.errors import TypingError
from numba.cuda.extending import intrinsic

from .declarations import ArrayOf, Declared, Record, Returns, Signature, is_scalar, numba_type
from .reading import Function

# One intrinsic per checked array, each naming the parameter it checks, gathered where a
# rewritten function reaches them through a single global.
_EXPECTATIONS = ModuleType("patos.cuda.typed.expectations")
_EXPECTED = itertools.count()


class Rewrite:
    """One function read by `Function`, rebuilt so its annotations convert what they declare."""

    def __init__(self, function: Function) -> None:
        self.function = function

    def rebuilt(self) -> FunctionType:
        """The function with every conversion in place and its annotations dropped.

        It carries the signature its callers' checks read, as `device_signature`.
        """
        original = self.function.function
        # The module's own globals, updated in place, so a name the module defines later resolves.
        namespace = self.function.namespace
        namespace.update(_patos_numpy=np, _patos_expectations=_EXPECTATIONS)
        rebuilt = FunctionType(self._code(), namespace, original.__name__)
        rebuilt.__doc__ = original.__doc__
        rebuilt.__qualname__ = original.__qualname__
        rebuilt.__dict__["device_signature"] = Signature(
            tuple(self.function.parameters.values()), self.function.returns
        )
        return rebuilt

    @staticmethod
    def _expectation(where: str) -> str:
        """Register an intrinsic that checks the element type of one array, returning its name.

        The intrinsic compiles to nothing when the array holds the scalar type its second
        argument names, and fails typing naming `where` otherwise.
        """

        @intrinsic
        def expects(_context, array: types.Type, kind: types.NumberClass) -> tuple:
            if getattr(array, "dtype", None) != kind.instance_type:
                raise TypingError(
                    f"{where} receives {array} where an Array of {kind.instance_type} is declared"
                )

            def lowering(context, _builder, _signature, _args):
                return context.get_dummy_value()

            return types.none(array, kind), lowering

        name = f"check_{next(_EXPECTED)}"
        setattr(_EXPECTATIONS, name, expects)
        return name

    def _code(self) -> CodeType:
        """The code object of the rewritten definition, compiled where the original was."""
        definition = self._rewritten()
        ast.fix_missing_locations(self.function.tree)
        module = compile(self.function.tree, self.function.function.__code__.co_filename, "exec")
        return next(
            constant
            for constant in module.co_consts
            if getattr(constant, "co_name", None) == definition.name
        )

    def _entry(self) -> list[ast.stmt]:
        """Return a check of every array's element type, then a conversion of every scalar.

        Tuple parameters convert element by element.
        """
        statements: list[ast.stmt] = []
        for argument in self.function.definition.args.args:
            declared = self.function.parameters[argument.arg]
            if not self.function.kernel and numba_type(declared) is not None:
                continue
            name = ast.Name(id=argument.arg, ctx=ast.Load())
            checks: list[ast.stmt] = []
            value = self._received(declared, name, checks)
            if value is not name:
                checks.append(
                    ast.Assign(targets=[ast.Name(id=argument.arg, ctx=ast.Store())], value=value)
                )
            statements.extend(ast.copy_location(statement, argument) for statement in checks)
        return statements

    def _expected(self, declared: ArrayOf, value: ast.expr) -> list[ast.stmt]:
        """The check that array `value` holds the element `declared` names, when it names one."""
        if declared.element is None:
            return []
        where = f"{self.function.function.__qualname__}'s `{ast.unparse(value)}`"
        expects = ast.Attribute(
            value=ast.Name(id="_patos_expectations", ctx=ast.Load()),
            attr=self._expectation(where),
            ctx=ast.Load(),
        )
        return [ast.Expr(value=ast.Call(func=expects, args=[value, _numpy(declared.element)]))]

    def _received(self, declared: Declared, value: ast.expr, checks: list[ast.stmt]) -> ast.expr:
        """`value` converted to what `declared` names, appending a check of each array in it.

        A record arrives with its scalars already converted where the host built it, so only its
        arrays are checked, field by field, and the record passes on unchanged.
        """
        if isinstance(declared, Record):
            for name, kind in declared.fields:
                field = ast.Attribute(value=value, attr=name, ctx=ast.Load())
                self._received(kind if not is_scalar(kind) else None, field, checks)
            return value
        if is_scalar(declared):
            return _converted(declared, value)
        if isinstance(declared, ArrayOf):
            checks.extend(self._expected(declared, value))
            return value
        if isinstance(declared, tuple):
            return self._unpacked(declared, value, checks)
        return value

    def _rewritten(self) -> ast.FunctionDef:
        """The definition with its annotations converting, and then dropped."""
        definition = _Conversions(self.function).visit(self.function.definition)
        docstring = definition.body[:1] if ast.get_docstring(definition) is not None else []
        definition.body = [*docstring, *self._entry(), *definition.body[len(docstring) :]] or [
            ast.Pass()
        ]
        definition.decorator_list = []
        definition.type_params = []
        definition.returns = None
        for argument in definition.args.args:
            argument.annotation = None
        return definition

    def _unpacked(
        self, declared: tuple[Declared, ...], value: ast.expr, checks: list[ast.stmt]
    ) -> ast.expr:
        """Tuple `value` received element by element, rebuilt only where an element converts."""
        parts = [
            ast.Subscript(value=value, slice=ast.Constant(value=index), ctx=ast.Load())
            for index in range(len(declared))
        ]
        received = [
            self._received(kind, part, checks) for kind, part in zip(declared, parts, strict=True)
        ]
        if any(element is not part for element, part in zip(received, parts, strict=True)):
            return ast.Tuple(elts=received, ctx=ast.Load())
        return value


class _Conversions(ast.NodeTransformer):
    """Rewrite one function so every return and every assignment to a declared local converts."""

    def __init__(self, function: Function) -> None:
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
