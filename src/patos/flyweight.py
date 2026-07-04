from abc import ABCMeta
from typing import TypeVar, cast

R = TypeVar("R")


class Arg:
    """One flyweight key element: hashes by its argument and compares without a tensor `__eq__`.

    The flyweight interns by argument value, but an argument whose `==` does not return a plain
    `bool` -- a pydantic model carrying tensors reduces a tensor to a bool and raises, a raw tensor
    returns an elementwise mask -- cannot key a plain `dict`, whose lookup would raise on the
    collision compare. Resolving a hash collision by identity first, then a guarded value compare
    that treats an unresolvable equality as distinct, keeps interning exact for well-behaved
    arguments and degrades to per-object for one whose equality is undefined, never crashing.
    Typing stays like `lru_cache(typed=True)`: the argument's type joins the hash and gates the
    compare, so `Arg(1)`, `Arg(True)` and `Arg(1.0)` are distinct keys.
    """

    __slots__ = ("kind", "value")

    def __init__(self, value: object) -> None:
        self.kind = type(value)
        self.value = value

    def __hash__(self) -> int:
        return hash((self.kind, self.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Arg) or self.kind is not other.kind:
            return False
        if self.value is other.value:
            return True
        try:
            return bool(self.value == other.value)
        except RuntimeError, ValueError, TypeError:
            return False


CacheKey = tuple[tuple[Arg, ...], frozenset[tuple[str, Arg]]]


class FlyweightMeta(ABCMeta):
    """Metaclass that interns instances by their construction arguments.

    The first construction with a given `(args, kwargs)` builds and caches the instance; every
    later construction with equal arguments returns that same object without re-running
    `__init__` -- the metaclass owns `__call__`, so no re-entrancy guard is needed (unlike
    caching `__new__`, which still re-runs `__init__`). Each class keeps its own cache and
    arguments must be hashable. Interning is typed like `lru_cache(typed=True)`, so `Node(1)`,
    `Node(True)` and `Node(1.0)` stay distinct instances even though the arguments compare
    equal. Subclasses `ABCMeta` so a flyweight can also be an abstract base or carry abstract
    methods.
    """

    # `cls: type[R]` makes `Node(...)` return `Node`; construction args stay `object` because
    # one metaclass serves every class, each with its own `__init__` signature.
    def __call__(cls: type[R], *args: object, **kwargs: object) -> R:
        cache: dict[CacheKey, object] = cls.__dict__.get("flyweights", {})
        if "flyweights" not in cls.__dict__:
            cls.flyweights = cache
        key = (
            tuple(Arg(arg) for arg in args),
            frozenset((name, Arg(value)) for name, value in kwargs.items()),
        )
        if key not in cache:
            cache[key] = type.__call__(cls, *args, **kwargs)
        return cast(R, cache[key])
