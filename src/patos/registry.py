from typing import ClassVar, Self, cast


class Registry:
    """Mixin for self-registering class hierarchies with try-each dispatch.

    Each direct child of `Registry` becomes a registry root that owns the list of its
    own concrete subclasses. `registry()` lists every enrolled class (the root included).
    `implementations()` is the concrete-only view callers usually want, `find(name)` looks
    one up by its `name`, and `dispatch` tries each implementation's `from_dispatch` until
    one succeeds.
    """

    registry_entries: ClassVar[list[type["Registry"]]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register `cls` as a root (if a direct child) and into its nearest root."""
        super().__init_subclass__(**kwargs)

        if Registry in cls.__bases__:
            cls.registry_entries = []
        cls.registry().append(cls)

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
    def find(cls, name: str, *, attr: str = "name") -> type[Self]:
        """Return the concrete implementation whose `attr` equals `name`.

        The typed replacement for the `{c.name: c for c in Base.registry()}[name]` lookup that
        keyed registries hand-roll. Matches against each implementation's `attr` value (default
        the `name` class attribute) and raises a `KeyError` listing the known keys when missing.

        name: the key to look up.
        attr: the class attribute carrying each implementation's key.
        """
        candidates = cls.implementations()
        matches = {getattr(impl, attr): impl for impl in candidates if hasattr(impl, attr)}
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

        On a registry root, every concrete implementation is tried in order and the last raised
        error is re-raised if all fail. Off a root, `cls` dispatches directly.
        """
        if not cls.is_registry_root():
            return cls.from_dispatch(*args, **kwargs)

        last_error: Exception | None = None
        for subclass in cls.implementations():
            try:
                return subclass.from_dispatch(*args, **kwargs)
            except Exception as error:
                last_error = error

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No implementation found in {cls.__name__} registry.")

    @classmethod
    def from_dispatch(cls, *args: object, **kwargs: object) -> Self:
        """Build an instance from dispatch arguments; subclasses override this."""
        raise NotImplementedError(f"{cls.__name__} does not implement from_dispatch.")
