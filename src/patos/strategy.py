from collections.abc import Callable, Iterator
from functools import partial
from typing import NamedTuple, Protocol, Self, cast, runtime_checkable

from .registry import Registry


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


class Resolution[T](NamedTuple):
    """The outcome of a cascade walk: who won and why the others did not.

    The rejection log is a value rather than a log line, so a caller can
    render why this host is on its fallback (`gold: port-22 ssh timed out,
    won: tailnet`) instead of silently degrading.
    """

    winner: str
    implementation: object
    rejected: tuple[tuple[str, str], ...]


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

    @classmethod
    def from_registry(
        cls,
        root: type[Registry],
        name: str = "strategy",
        **factory_kwargs: object,
    ) -> Self:
        """A strategy whose lazy factories are `root`'s concrete implementations, keyed by name.

        The bridge between the two halves of the pattern. `Registry` collects the concrete
        classes as they are imported and `Strategy` picks one of them by name at runtime, so a
        consumer holding a registry root gets the whole named family in one call rather than
        hand rolling the `for impl in Root.implementations()` loop once per family. Every
        implementation stays lazy, so only the one actually selected is ever constructed, which
        is what keeps a family of model backed stages cheap to enumerate.

        root: the registry root whose `implementations()` become the registrations, in
            registration order, so `first_available` and `cascade` walk them by preference.
        name: human readable label used in error messages.
        factory_kwargs: passed to every implementation's constructor when it is first selected,
            which is how one shared settings object reaches an entire family.
        """
        strategy = cls(name)
        for implementation in root.implementations():
            build = cast(Callable[[], T], partial(implementation, **factory_kwargs))
            strategy.factory(implementation.name, build)
        return strategy

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

    def cascade(self) -> Resolution[T]:
        """Walk registrations in order, returning the winner with the rejection log.

        `first_available` with receipts: every impl passed over is recorded as
        `(name, reason)`, where the reason is the probe's exception when it
        raised and a plain report when it returned falsy. All-rejected raises
        `StrategyError` carrying every reason, so the failure names each link.
        """
        rejected: list[tuple[str, str]] = []
        for name in self.factories:
            impl = self.resolve(name)
            availability = getattr(impl, "available", True)
            try:
                if callable(availability):
                    availability = availability()
            except Exception as error:
                rejected.append((name, f"{type(error).__name__}: {error}"))
                continue
            if availability:
                return Resolution(winner=name, implementation=impl, rejected=tuple(rejected))
            rejected.append((name, "reported unavailable"))
        reasons = "; ".join(f"{name}: {reason}" for name, reason in rejected)
        raise StrategyError(f"{self.name}: every implementation refused, {reasons}")

    @property
    def names(self) -> list[str]:
        """Registered names in insertion order."""
        return list(self.factories)

    def __contains__(self, name: str) -> bool:
        return name in self.factories

    def __len__(self) -> int:
        return len(self.factories)

    def __iter__(self) -> Iterator[str]:
        return iter(self.factories)

    def __repr__(self) -> str:
        return f"<Strategy {self.name!r} names={list(self.factories)}>"
