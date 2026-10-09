"""The names a device function reads, in the order Python reads them.

Its module's globals stay live, so a name the module defines later resolves. A scope it closes
over comes first, and a name that scope has not bound yet is None, never a global of the same
name. A local shadows a global of its name, and one bound once to a class, function or module
stands for it. The function rebuilt from a rewrite keeps those scopes' own cells.
"""

import ast
from collections import ChainMap, Counter
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from types import CellType, CodeType, FunctionType

import numpy as np

from .declarations import Evaluated, unaliased

# A cell no value is stored in yet; cells compare by what they hold, an empty one equal to it.
_EMPTY = CellType()


def body_scope(
    function: FunctionType, nodes: Iterable[ast.AST], arguments: Iterable[ast.arg]
) -> ChainMap[str, Evaluated]:
    """What the names in the body `nodes` of `function` resolve to.

    A local assigned once to a plain or dotted name resolves to what that name does if it is a
    class, function or module: `Alias = module.Bounds` makes `Alias(...)` build a `Bounds`.
    """
    ordered = sorted(nodes, key=lambda node: getattr(node, "lineno", 0))
    stores = Counter(
        node.id
        for node in ordered
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    stores.update(argument.arg for argument in arguments)
    scope = ChainMap(dict.fromkeys(stores), _cells(function), function.__globals__)
    for node in ordered:
        if (alias := _alias(node, scope, stores)) is not None:
            scope[alias[0]] = alias[1]
    return scope


def annotation_scope(function: FunctionType, owner: type | None) -> ChainMap[str, Evaluated]:
    """What a name in an annotation of `function` resolves to before the module's names.

    The class body the function was written in (`__classdict__`, PEP 649), then what its
    `__annotate__` and the function itself close over, then the record it is a member of, which
    the module does not name yet while the record is being made.
    """
    annotate = function.__annotate__
    closed = _cells(annotate) if isinstance(annotate, FunctionType) else {}
    classdict = closed.pop("__classdict__", None)
    maps: Sequence[MutableMapping[str, Evaluated]] = [
        classdict if isinstance(classdict, dict) else {},
        closed,
        _cells(function),
        {} if owner is None else {owner.__name__: owner},
    ]
    return ChainMap(*maps)


def resolved(node: ast.expr, namespace: Mapping[str, Evaluated]) -> Evaluated:
    """Return what a plain or dotted name refers to in `namespace`, through any type alias.

    Anything else resolves to None.
    """
    value: Evaluated = None
    if isinstance(node, ast.Name):
        value = namespace.get(node.id)
    elif isinstance(node, ast.Attribute):
        owner = resolved(node.value, namespace)
        value = getattr(owner, node.attr, None) if owner is not None else None
    return unaliased(value)


def scoped(definition: ast.FunctionDef, original: FunctionType) -> CodeType:
    """`definition` compiled in a scope that binds the names `original` closes over.

    They stay free in it, as they are in `original`.
    """
    names = original.__code__.co_freevars
    closed = [ast.Assign([ast.Name(n, ast.Store())], ast.Constant(None)) for n in names]
    scope = ast.FunctionDef("_patos_scope", ast.arguments(), [*closed, definition])
    module = ast.Module([ast.copy_location(scope, definition)])
    code = compile(ast.fix_missing_locations(module), original.__code__.co_filename, "exec")
    return _nested(_nested(code, scope.name), definition.name)


def closure_of(original: FunctionType, code: CodeType) -> tuple[CellType, ...] | None:
    """The cells of `original`'s scopes that `code` reads, the scopes' own.

    A name a scope binds after the function is made reaches the function built on them.
    """
    cells = dict(zip(original.__code__.co_freevars, original.__closure__ or (), strict=True))
    return (*(cells[name] for name in code.co_freevars),) or None


def _cells(function: FunctionType) -> dict[str, Evaluated]:
    """What `function` closes over, by name.

    A cell its scope has not filled yet holds None, since that scope is the one to name it later.
    """
    if not function.__closure__:
        return {}
    pairs = zip(function.__code__.co_freevars, function.__closure__, strict=True)
    return {name: None if cell == _EMPTY else cell.cell_contents for name, cell in pairs}


def _nested(code: CodeType, name: str) -> CodeType:
    """The code object of the function `name` that `code` defines."""
    (found,) = (c for c in code.co_consts if getattr(c, "co_name", None) == name)
    return found


def _alias(
    node: ast.AST, scope: Mapping[str, Evaluated], stores: Counter[str]
) -> tuple[str, Evaluated] | None:
    """The local `node` assigns once to a plain or dotted name, and what that name resolves to.

    Only a class, function or module counts, not a constant.
    """
    match node:
        case ast.Assign(
            targets=[ast.Name(id=name)], value=ast.Name() | ast.Attribute() as value
        ) if stores[name] == 1:
            found = resolved(value, scope)
            if found is not None and not isinstance(found, int | np.generic):
                return name, found
    return None
