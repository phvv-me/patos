import re
from typing import ClassVar, Self, cast

CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class Registry:
    """Mixin for self-registering class hierarchies with try-each dispatch.

    Each direct child of `Registry` becomes a registry root that owns the list of its
    own concrete subclasses. `registry()` lists every enrolled class (the root included).
    `implementations()` is the concrete-only view callers usually want, `find(name)` looks
    one up by its `name`, and `dispatch` tries each implementation's `from_dispatch` until
    one succeeds.
    """

    registry_entries: ClassVar[list[type["Registry"]]]
    name: ClassVar[str]  # auto-derived kebab-case of the class name unless declared

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register `cls` as a root (if a direct child) and into every root it inherits.

        A class deriving from two registry roots enrolls in both, so each root's
        `registry()` sees it. Every subclass also receives an auto-derived kebab-case
        `name` from its class name (`OpusFB8` becomes `opus-fb8`) unless the class body
        declares one itself.
        """
        super().__init_subclass__(**kwargs)

        if "name" not in cls.__dict__ and "name" not in cls.__annotations__:
            cls.name = CAMEL_BOUNDARY.sub("-", cls.__name__).lower()
        if Registry in cls.__bases__:
            cls.registry_entries = []
        for base in cls.__mro__:
            if "registry_entries" in base.__dict__:
                cast(type[Registry], base).registry_entries.append(cls)

    @classmethod
    def root(cls) -> type[Self]:
        """Return the registry root that owns this class's implementation list."""
        for base in cls.__mro__:
            if "registry_entries" in base.__dict__:
                return cast(type[Self], base)
        raise RuntimeError(f"No registry found for {cls.__name__}.")

    @classmethod
    def registry(cls) -> list[type[Self]]:
        """Return the implementation list owned by this class's nearest registry root."""
        return cast(list[type[Self]], cls.root().registry_entries)

    @classmethod
    def is_registry_root(cls) -> bool:
        """Return whether this class owns a registry."""
        return "registry_entries" in cls.__dict__

    @classmethod
    def implementations(cls) -> list[type[Self]]:
        """Concrete enrolled classes, excluding the registry root and any abstract bases.

        The view consumers reach for to fan out over real providers: it drops the root itself
        and any class still carrying abstract methods, so `for impl in Base.implementations()`
        replaces the hand-rolled `for c in Base.registry() if c is not Base and ...` filter.
        """
        root = cls.root()
        return [
            entry
            for entry in cls.registry()
            if entry is not root and not getattr(entry, "__abstractmethods__", frozenset())
        ]

    @classmethod
    def names(cls) -> list[str]:
        """The `name` key of every concrete implementation, the keys `find` accepts.

        The companion to `implementations()` for the common `[impl.name for impl in
        Base.implementations()]` listing, so callers enumerate the registry's keys without
        reaching through each class.
        """
        return [impl.name for impl in cls.implementations()]

    @classmethod
    def find(cls, name: str, *, attr: str = "name") -> type[Self]:
        """Return the concrete implementation whose own `attr` equals `name`.

        The typed replacement for the `{c.name: c for c in Base.registry()}[name]` lookup that
        keyed registries hand-roll. Only attributes defined on the implementation class itself
        count, so a subclass that forgot its own key never answers for an inherited one, and a
        duplicate key raises a `ValueError` instead of silently shadowing an earlier class. A
        miss raises a `KeyError` listing the known keys.

        name: the key to look up.
        attr: the class attribute carrying each implementation's key.
        """
        matches: dict[object, type[Self]] = {}
        for impl in cls.implementations():
            if attr not in vars(impl):
                continue
            key = vars(impl)[attr]
            if key in matches:
                raise ValueError(
                    f"{cls.__name__} has duplicate {attr}={key!r} on "
                    f"{matches[key].__name__} and {impl.__name__}.",
                )
            matches[key] = impl
        try:
            return matches[name]
        except KeyError:
            known = sorted(map(repr, matches))
            raise KeyError(
                f"{cls.__name__} has no implementation with {attr}={name!r}. "
                f"Known {attr}s are {known}.",
            ) from None

    @classmethod
    def dispatch(cls, *args: object, **kwargs: object) -> Self:
        """Try each registered implementation's `from_dispatch`, returning the first success.

        On a registry root, every concrete implementation is tried in order and, when all
        fail, their refusals are raised together as an `ExceptionGroup` so no error is
        lost. Off a root, `cls` dispatches directly.
        """
        if not cls.is_registry_root():
            return cls.from_dispatch(*args, **kwargs)

        errors: list[Exception] = []
        for subclass in cls.implementations():
            try:
                return subclass.from_dispatch(*args, **kwargs)
            except Exception as error:
                errors.append(error)

        if errors:
            raise ExceptionGroup(
                f"every implementation in the {cls.__name__} registry refused dispatch",
                errors,
            )
        raise RuntimeError(f"No implementation found in {cls.__name__} registry.")

    @classmethod
    def from_dispatch(cls, *args: object, **kwargs: object) -> Self:
        """Build an instance from dispatch arguments; subclasses override this."""
        raise NotImplementedError(f"{cls.__name__} does not implement from_dispatch.")
