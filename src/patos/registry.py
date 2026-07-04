import re
from collections.abc import Callable
from typing import ClassVar, Self, cast

# Split a camel name only where a new *word* starts: at a lowercase run meeting an upper
# (`opusFb8` -> `opus-fb8`), and before a capitalised word however its left neighbour reads, so
# both an acronym run (`HTTPServer` -> `http-server`) and a digit run (`E8Lattice` -> `e8-lattice`)
# cut before the word. A bare capital that merely ends an acronym token never starts a word, so a
# pure acronym keeps one segment whether or not it carries a digit (`RVQ` -> `rvq`, `E8P` ->
# `e8p`), which is what makes the kebab key idempotent and the find round-trip stable.
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[a-zA-Z0-9])(?=[A-Z][a-z])")


def generic_alias(impl: type) -> bool:
    """Whether `impl` is a pydantic generic parametrization, not a distinct implementation.

    Subclassing a generic model under PEP 695 (`class Sub[C](Base[C])`) makes pydantic
    materialize one intermediate class per concrete parametrization a subclass pins
    (`Base[Tensor]`, `Base[tuple[Tensor, C]]`), and each trips `__init_subclass__` and enrolls
    in the registry under a bracketed kebab name (`base[tensor]`). Those aliases are typing
    artifacts of their `origin` class, never separate providers, so `implementations()` drops
    them. The signal is pydantic's own `__pydantic_generic_metadata__["origin"]`, which the alias
    carries and the real concrete class leaves `None`; a non-pydantic class has no such attribute
    and is kept. This keeps `names()` and `find` to the genuine implementations.

    impl: the registry member being classified.
    """
    return getattr(impl, "__pydantic_generic_metadata__", {}).get("origin") is not None


def available(impl: type) -> bool:
    """Whether an implementation *class* reports itself runnable on this host.

    The default availability probe `Registry.first_available` walks with. It reads a class-level
    `is_available()` (mainboard's tracer convention) or, failing that, an `available()` the class
    exposes (a classmethod or staticmethod), calling whichever it finds. A class declaring neither
    is treated as always available, so a plain fallback needs no boilerplate to be the catch-all.
    A bound instance method (atpx's `Engine.available(self)`) is not callable on the bare class, so
    such a consumer passes its own `probe` to `first_available` instead of relying on this default.

    impl: the implementation class being probed.
    """
    for name in ("is_available", "available"):
        candidate = getattr(impl, name, None)
        if callable(candidate):
            return bool(candidate())
    return True


class Registry:
    """Mixin for self-registering class hierarchies with try-each dispatch.

    Each direct child of `Registry` becomes a registry root that owns the list of its
    own concrete subclasses. `registry()` lists every enrolled class (the root included).
    `implementations()` is the concrete-only view callers usually want, `find(name)` looks
    one up by its `name`, and `dispatch` tries each implementation's `from_dispatch` until
    one succeeds.
    """

    registry_entries: ClassVar[list[type[Registry]]]
    name: ClassVar[str]  # auto-derived kebab-case of the class name unless declared

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register `cls` as a root (if a direct child) and into every root it inherits.

        A class deriving from two registry roots enrolls in both, so each root's
        `registry()` sees it. Every subclass also receives an auto-derived kebab-case
        `name` from its class name (`OpusFB8` becomes `opus-fb8`) unless the class body
        declares one itself.
        """
        super().__init_subclass__(**kwargs)

        if "name" not in cls.__dict__:
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
        """Concrete subclasses of `cls`, excluding the registry root and any abstract bases.

        The view consumers reach for to fan out over real providers: it drops the root itself
        and any class still carrying abstract methods, so `for impl in Base.implementations()`
        replaces the hand-rolled `for c in Base.registry() if c is not Base and ...` filter.

        The view is scoped to `cls`'s own subtree, not the whole shared root: when two sibling
        hierarchies share one registry root (the codec `Quantizer` and the `Lattice` oracle both
        enroll under `Component`), `Lattice.implementations()` lists only lattices, never the
        quantizers that happen to share the root. Without that scope `Lattice.find` would scan
        every quantizer too, and two same-named classes reachable through the root (a re-imported
        module under a test runner) would collide on a key neither hierarchy actually shares.
        """
        return [
            entry
            for entry in cls.registry()
            if entry is not cls
            and issubclass(entry, cls)
            and not getattr(entry, "__abstractmethods__", frozenset())
            and not generic_alias(entry)
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
    def find(cls, name: str, *, attr: str = "name", default: str | None = None) -> type[Self]:
        """Return the concrete implementation whose own `attr` equals `name`.

        The typed replacement for the `{c.name: c for c in Base.registry()}[name]` lookup that
        keyed registries hand-roll. Only attributes defined on the implementation class itself
        count, so a subclass that forgot its own key never answers for an inherited one, and a
        duplicate key raises a `ValueError` instead of silently shadowing an earlier class. A
        miss falls back to `default` when one is given and registered, so a caller that wants a
        graceful fallback opts in explicitly; with no `default`, or a `default` that is itself
        unregistered, a miss raises a `KeyError` listing the known keys.

        name: the key to look up.
        attr: the class attribute carrying each implementation's key.
        default: key to fall back to when `name` is not registered.
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
        if name in matches:
            return matches[name]
        if default is not None and default in matches:
            return matches[default]
        known = sorted(map(repr, matches))
        raise KeyError(
            f"{cls.__name__} has no implementation with {attr}={name!r}. "
            f"Known {attr}s are {known}.",
        ) from None

    @classmethod
    def select(cls, predicate: Callable[[type[Self]], bool]) -> list[type[Self]]:
        """Concrete implementations satisfying `predicate`, in registration (preference) order.

        The typed replacement for the `[impl for impl in Base.implementations() if ...]` filter
        that keyed registries hand-roll to fan out over a subset (engines serving one capability,
        backends matching a vendor). The order is the registration order `implementations()`
        already fixes, so the first match is the preferred one.

        predicate: keeps an implementation when it returns true for that class.
        """
        return [impl for impl in cls.implementations() if predicate(impl)]

    @classmethod
    def first_available(
        cls,
        probe: Callable[[type[Self]], bool] = lambda impl: available(impl),
    ) -> type[Self]:
        """The first concrete implementation whose availability `probe` passes, raising on none.

        The "pick the first thing that works" selection mainboard's tracer detect and atpx's
        engine choice both hand-roll: walk `implementations()` in registration (preference) order
        and return the first the host can actually run. The default `probe` reads an `available()`
        or `is_available()` method (the `Available` convention `Strategy` shares), counting an
        implementation without one as always available, so a plain fallback registered last is the
        catch-all. Pass a `probe` to key availability on anything else.

        probe: returns whether an implementation may be chosen on this host.
        """
        for impl in cls.implementations():
            if probe(impl):
                return impl
        raise LookupError(
            f"{cls.__name__} has no available implementation among {cls.names()}.",
        )

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
