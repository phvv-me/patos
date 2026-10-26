"""The signatures a dispatched name declares and the implementation that runs each, read and
checked at import.

A stub declares as many signatures as it has `typing.overload`s, or one, a constrained type
parameter standing in turn for each of its constraints and all the parameters of one for the
same. Each is run by the implementation that declares its types, or else the one generic over
them, and returns what the stub declares.
"""

import inspect
import itertools
import math
import operator
from collections.abc import Callable, Iterator, Sequence
from functools import cache
from types import FunctionType, NoneType
from typing import TypeVar, get_overloads

from .declarations import Declared, Evaluated, Returns, Signature, declared, named

# The parameter types of one signature a dispatched name takes.
type Kinds = tuple[Declared, ...]
# The most signatures a stub declares: independent type parameters multiply, and eight over six
# types would be 1.7 million.
_MOST_SIGNATURES = 1024


def listed(kinds: Kinds) -> str:
    """The parameter types of a signature as a tuple of patos's short names."""
    return f"({', '.join(map(named, kinds))})"


def taken(stub: FunctionType) -> dict[Kinds, Returns]:
    """The return type of every signature `stub` declares, by its parameter types.

    Raises `TypeError` for an overload of another arity than the stub, a type parameter without
    constraints, more signatures than are listed, a type that is no device type, or a signature
    declared twice.
    """
    name, arity = stub.__qualname__, len(inspect.signature(stub).parameters)
    found: dict[Kinds, Returns] = {}
    for declaring in get_overloads(stub) or [stub]:
        for kinds, returns in _expanded(name, arity, inspect.signature(declaring)):
            if kinds in found:
                raise TypeError(f"{name}: declares {listed(kinds)} twice")
            found[kinds] = returns
    return found


def implementing(
    name: str, kinds: Kinds, returns: Returns, implementations: Sequence[Callable]
) -> Callable:
    """The implementation that declares `kinds`, or else the one generic where it does not.

    It returns `returns`, which the stub declares for `kinds`.
    """
    fitting = {
        implementation: sum(map(operator.eq, own.parameters, kinds, strict=True))
        for implementation in implementations
        if _is_taken(own := _declared_by(implementation), kinds)
    }
    best = [found for found, count in fitting.items() if count == max(fitting.values())]
    if len(best) != 1:
        raise TypeError(f"{name}: {len(best) or 'no'} implementations take {listed(kinds)}")
    if not _has_return(_declared_by(best[0]), kinds, returns):
        raise TypeError(
            f"{name}: {best[0]!r} does not return {named(returns)} for {listed(kinds)}"
        )
    return best[0]


def _declared_by(implementation: Callable) -> Signature:
    """What a device function or a PTX stub declares, None where generic."""
    compiled = getattr(implementation, "py_func", implementation)
    signature: Signature | None = getattr(compiled, "device_signature", None)
    if signature is None:
        raise TypeError(f"{implementation!r} is no device function patos typed")
    return signature


def _expanded(name: str, arity: int, read: inspect.Signature) -> Iterator[tuple[Kinds, Returns]]:
    """Each signature `read` stands for, a type parameter in turn each of its constraints."""
    annotations = [parameter.annotation for parameter in read.parameters.values()]
    if len(annotations) != arity:
        raise TypeError(
            f"{name}: an overload takes {len(annotations)} parameters, the stub {arity}"
        )
    generic = _generic(name, annotations)
    result = read.return_annotation
    for chosen in itertools.product(*(variable.__constraints__ for variable in generic)):
        bound = dict(zip(generic, chosen, strict=True))
        kinds = (*(declared(bound.get(kind, kind)) for kind in annotations),)
        returns = NoneType if result is None else declared(bound.get(result, result))
        if None in kinds or returns is None:
            raise TypeError(f"{name}: {read} declares a type that is no device type")
        yield kinds, returns


def _generic(name: str, annotations: Sequence[Evaluated]) -> list[TypeVar]:
    """The type parameters among `annotations`, each with constraints and few enough in all."""
    generic = list(dict.fromkeys(kind for kind in annotations if isinstance(kind, TypeVar)))
    if bare := [variable for variable in generic if not variable.__constraints__]:
        raise TypeError(f"{name}: {bare[0]} lists no constraints, so no signatures")
    if math.prod(len(variable.__constraints__) for variable in generic) > _MOST_SIGNATURES:
        raise TypeError(f"{name}: declares more than {_MOST_SIGNATURES} signatures")
    return generic


def _is_taken(own: Signature, kinds: Kinds) -> bool:
    """Whether the implementation `own` takes `kinds`.

    A parameter declares its kind, or is generic over a type parameter whose constraints hold it,
    the same kind for all the parameters of one type parameter.
    """
    if len(own.parameters) != len(kinds):
        return False
    variables, bound = dict(own.variables), {}
    for index, (mine, kind) in enumerate(zip(own.parameters, kinds, strict=True)):
        if (variable := variables.get(index)) is None:
            fits = mine is None or mine == kind
        else:
            ranges = _constraints(variable)
            fits = (not ranges or kind in ranges) and bound.setdefault(variable, kind) == kind
        if not fits:
            return False
    return True


def _has_return(own: Signature, kinds: Kinds, returns: Returns) -> bool:
    """Whether the implementation `own` returns `returns` when it takes `kinds`.

    It declares the type, or returns the type parameter the operands bind to it.
    """
    variables = dict(own.variables)
    if (variable := variables.get(len(own.parameters))) is None:
        return own.returns is None or own.returns == returns
    bound = [kind for index, kind in enumerate(kinds) if variables.get(index) is variable]
    return not bound or bound[0] == returns


@cache
def _constraints(variable: TypeVar) -> list[Declared]:
    """The types a type parameter ranges over, as device code declares them."""
    return [declared(constraint) for constraint in variable.__constraints__]
