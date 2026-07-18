from typing import TypeVar, cast

R = TypeVar("R")


class SingletonMeta(type):
    """Metaclass that yields one instance per class, building it at most once.

    Owning `__call__` means the cached instance is returned without re-running `__init__`,
    avoiding the wart of caching `__new__` (which re-initialises on every call). The instance
    is stored on the class itself (mirroring `FlyweightMeta`'s per-class cache), so subclasses
    are independent singletons and no global table pins classes alive.
    """

    # `cls: type[R]` makes `Counter()` return `Counter`; construction args stay `object`
    # because one metaclass serves every class, each with its own `__init__` signature.
    def __call__(cls: type[R], *args: object, **kwargs: object) -> R:
        if "singleton_instance" not in cls.__dict__:
            type.__setattr__(cls, "singleton_instance", type.__call__(cls, *args, **kwargs))
        return cast(R, cls.__dict__["singleton_instance"])


class Singleton(metaclass=SingletonMeta):
    """Base for single-instance classes: every construction returns the same object.

    `__init__` runs only on the first construction, so later calls never re-initialise the
    shared instance. The no-argument degenerate case of the flyweight pattern.
    """
