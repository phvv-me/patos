from typing import Any, ClassVar, Self, cast


class Registry:
    """Mixin for self-registering class hierarchies with try-each dispatch.

    Each direct child of `Registry` becomes a registry root that owns the list of its
    own concrete subclasses. `registry()` lists implementations; `dispatch` tries each
    registered implementation's `from_dispatch` until one succeeds.
    """

    registry_entries: ClassVar[list[type["Registry"]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register `cls` as a root (if a direct child) and into its nearest root."""
        super().__init_subclass__(**kwargs)

        if Registry in cls.__bases__:
            cls.registry_entries = []

        for base in cls.__mro__:
            if "registry_entries" in base.__dict__:
                base.registry_entries.append(cls)
                return

    @classmethod
    def registry(cls) -> list[type["Registry"]]:
        """Return the implementation list owned by this class's nearest registry root."""
        for base in cls.__mro__:
            if "registry_entries" in base.__dict__:
                return base.registry_entries
        raise RuntimeError(f"No registry found for {cls.__name__}.")

    @classmethod
    def is_registry_root(cls) -> bool:
        """Return whether this class owns a registry."""
        return "registry_entries" in cls.__dict__

    @classmethod
    def dispatch(cls, *args: Any, **kwargs: Any) -> Self:
        """Try each registered implementation's `from_dispatch`, returning the first success.

        On a registry root, every other registered implementation is tried in order and the
        last raised error is re-raised if all fail; off a root, `cls` dispatches directly.
        """
        if not cls.is_registry_root():
            return cls.from_dispatch(*args, **kwargs)

        last_error: Exception | None = None
        for subclass in cls.registry():
            if subclass is cls:
                continue
            try:
                return cast(Self, subclass.from_dispatch(*args, **kwargs))
            except Exception as error:
                last_error = error

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No implementation found in {cls.__name__} registry.")

    @classmethod
    def from_dispatch(cls, *args: Any, **kwargs: Any) -> Self:
        """Build an instance from dispatch arguments; subclasses override this."""
        raise NotImplementedError(f"{cls.__name__} does not implement from_dispatch.")
