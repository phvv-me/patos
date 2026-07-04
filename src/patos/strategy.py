from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class Available(Protocol):
    """An implementation that can report whether it applies on the current host.

    Implementations may expose `available()` for first-available selection. A plain
    `available` boolean attribute also works at runtime. Those exposing neither are
    treated as always available, so plain value objects need no boilerplate to
    participate in named selection.
    """

    def available(self) -> bool:
        """Whether this implementation should be chosen in `first_available` mode."""


class StrategyError(LookupError):
    """Raised when a name is missing with no default, or no impl is available.

    A `LookupError` so failed selection still reads as a lookup failure, without
    `KeyError`'s quoted-repr rendering mangling the message.
    """


class Strategy[T]:
    """A family of interchangeable named implementations with runtime selection.

    `Strategy` replaces the bespoke `{"a": A(), "b": B()}.get(kind, default)` and
    "pick the first thing that works" dispatch tables scattered across services.
    Implementations register by name, eagerly (a value) or lazily (a zero-arg
    factory called on first use and cached). Two selection modes:

    - `select(name, default=...)`: keyed lookup, like `dict.get` but raising a
      clear `StrategyError` instead of returning `None` when nothing matches.
    - `first_available()`: walk registrations in insertion order and return the
      first whose `available()` predicate is true (impls without one always are).

    It is generic over the implementation or Protocol type, so `Strategy[Scheduler]`
    keeps `select`/`first_available` returning `Scheduler`.

    name: human-readable label used in error messages.
    """

    def __init__(self, name: str = "strategy") -> None:
        self.name = name
        self.factories: dict[str, Callable[[], T]] = {}
        self.cache: dict[str, T] = {}

    def register(self, name: str, impl: T) -> None:
        """Register an already-built implementation value under `name`.

        impl: the implementation; stored as-is and returned by `select`/`first_available`.
        """
        self.cache[name] = impl
        self.factories[name] = lambda: impl

    def factory(self, name: str, build: Callable[[], T]) -> None:
        """Register a zero-arg factory under `name`, built lazily on first resolution.

        Re-registering a name drops any instance already cached for it, so the new
        factory wins on the next resolution.

        name: the implementation key.
        build: produces the implementation the first time it is selected; cached after.
        """
        self.cache.pop(name, None)
        self.factories[name] = build

    def add(self, name: str) -> Callable[[Callable[[], T]], Callable[[], T]]:
        """Decorator form of `factory`, registering the decorated zero-arg builder under `name`."""

        def decorate(build: Callable[[], T]) -> Callable[[], T]:
            self.factory(name, build)
            return build

        return decorate

    def resolve(self, name: str) -> T:
        """Build (once) and return the implementation registered under `name`."""
        try:
            return self.cache[name]
        except KeyError:
            self.cache[name] = self.factories[name]()
            return self.cache[name]

    def select(self, name: str, default: str | None = None) -> T:
        """Return the impl for `name`, or for `default`, raising if neither is registered.

        name: the requested implementation key.
        default: fallback key used when `name` is unknown; `None` means no fallback.
        """
        if name in self.factories:
            return self.resolve(name)
        if default is not None and default in self.factories:
            return self.resolve(default)
        raise StrategyError(
            f"{self.name}: no implementation for {name!r}; choose from {sorted(self.factories)}"
        )

    def first_available(self) -> T:
        """Return the first registered impl whose availability is true, in insertion order.

        Availability comes from the impl's `available` attribute, called when it is a
        method and taken as the truth value when it is plain data. Implementations
        without one count as always available, so a plain default placed last is the
        catch-all.
        """
        for name in self.factories:
            impl = self.resolve(name)
            availability = getattr(impl, "available", True)
            if callable(availability):
                availability = availability()
            if availability:
                return impl
        raise StrategyError(
            f"{self.name}: no available implementation among {sorted(self.factories)}"
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Registered names in insertion order."""
        return tuple(self.factories)

    def __contains__(self, name: str) -> bool:
        return name in self.factories

    def __len__(self) -> int:
        return len(self.factories)

    def __iter__(self) -> Iterator[str]:
        return iter(self.factories)

    def __repr__(self) -> str:
        return f"<Strategy {self.name!r} names={list(self.factories)}>"
