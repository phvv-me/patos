from typing import Any, ClassVar


class SingletonMeta(type):
    """Metaclass that yields one instance per class, building it at most once.

    Owning `__call__` means the cached instance is returned without re-running `__init__`,
    avoiding the wart of caching `__new__` (which re-initialises on every call). Each class
    keeps its own instance, so subclasses are independent singletons.
    """

    instances: ClassVar[dict[type, Any]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in SingletonMeta.instances:
            SingletonMeta.instances[cls] = super().__call__(*args, **kwargs)
        return SingletonMeta.instances[cls]


class Singleton(metaclass=SingletonMeta):
    """Base for single-instance classes: every construction returns the same object.

    `__init__` runs only on the first construction, so later calls never re-initialise the
    shared instance. The no-argument degenerate case of the flyweight pattern.
    """
