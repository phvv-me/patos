from abc import ABCMeta
from typing import TypeVar, cast

R = TypeVar("R")
CacheKey = tuple[tuple[object, ...], frozenset[tuple[str, object]]]


class FlyweightMeta(ABCMeta):
    """Metaclass that interns instances by their construction arguments.

    The first construction with a given `(args, kwargs)` builds and caches the instance; every
    later construction with equal arguments returns that same object without re-running
    `__init__` -- the metaclass owns `__call__`, so no re-entrancy guard is needed (unlike
    caching `__new__`, which still re-runs `__init__`). Each class keeps its own cache and
    arguments must be hashable. Subclasses `ABCMeta` so a flyweight can also be an abstract
    base or carry abstract methods.
    """

    # `cls: type[R]` makes `Node(...)` return `Node`; construction args stay `object` because
    # one metaclass serves every class, each with its own `__init__` signature.
    def __call__(cls: type[R], *args: object, **kwargs: object) -> R:
        cache: dict[CacheKey, object] = cls.__dict__.get("flyweights", {})
        if "flyweights" not in cls.__dict__:
            cls.flyweights = cache
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = type.__call__(cls, *args, **kwargs)
        return cast(R, cache[key])
