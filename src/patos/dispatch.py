import functools
import inspect
from collections.abc import Callable, Hashable
from types import MappingProxyType
from typing import TypeGuard, cast


class value_dispatch[**P, R]:
    """Turn a function into a value-dispatched generic.

    Like `functools.singledispatch`, but the dispatch key is the *value* of a
    keyword argument (default `"kind"`) rather than the *type* of the first
    positional argument. Both decorator forms work:

        @value_dispatch
        def render(node, **kw): ...        # bare: kind is "kind"

        @value_dispatch(kind="how")
        def render(node, **kw): ...        # kind is "how"

    Implementations register under a key with `@render.register()`. Each is also
    exposed as an attribute (`render.html`) so callers that already know the
    kind can skip the registry lookup. Calling with no kind (or `kind=None`)
    falls back to the originally decorated function.

    fallback: the wrapped function; supplied implicitly by the decorator.
    kind: name of the keyword argument carrying the dispatch value.
    """

    def __init__(
        self,
        fallback: Callable[P, R] | None = None,
        *,
        kind: str = "kind",
    ) -> None:
        self.kind_arg = kind
        self.registry_map: dict[Hashable, Callable[P, R]] = {}
        self.fallback: Callable[P, R] | None = None
        if fallback is not None:
            self.bind_fallback(fallback)

    def bind_fallback(self, fallback: Callable[P, R]) -> None:
        """Adopt `fallback` as the default impl and copy its metadata onto self.

        Rejects functions whose first parameter is `self`, since a dispatcher stored in
        a class body is not a descriptor and never binds as a method. `updated=()` keeps
        the fallback's `__dict__` from being copied over the dispatcher's own state.
        """
        if inspect.isfunction(fallback) and fallback.__code__.co_varnames[:1] == ("self",):
            raise TypeError(
                "value_dispatch does not support methods. Decorate a module-level "
                "function instead of one whose first parameter is self.",
            )
        self.fallback = fallback
        functools.update_wrapper(self, fallback, updated=())

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Bind the fallback on the parametrised first call, otherwise dispatch.

        With no fallback yet, this is the parametrised-decorator stage: the sole
        argument is the function to wrap. Once bound, pop the kind keyword and
        route to its registered impl, falling back when the kind is absent.
        """
        if self.fallback is None:
            if len(args) != 1 or kwargs or not callable(args[0]):
                raise TypeError(
                    "this value_dispatch has no function bound yet. Apply the "
                    "parametrised dispatcher as a decorator to its fallback function "
                    "before calling it.",
                )
            self.bind_fallback(cast("Callable[P, R]", args[0]))
            # The parametrised form `@value_dispatch(kind=...)` then `@dispatcher` returns
            # the dispatcher itself, so this branch yields self rather than an `R`. The cast
            # keeps both decorator stages behind one `__call__` without splitting the API.
            return cast(R, self)
        value = kwargs.pop(self.kind_arg, None)
        if value is None:
            return self.fallback(*args, **kwargs)
        try:
            impl = self.registry_map[value]
        except KeyError:
            raise ValueError(
                f"{self.fallback.__name__}: unknown {self.kind_arg}={value!r}; "
                f"choose from {sorted(self.registry_map, key=repr)}",
            ) from None
        return impl(*args, **kwargs)

    def register[**Q](
        self,
        key: Hashable | None = None,
        *,
        name: Hashable | None = None,
    ) -> Callable[[Callable[Q, R]], Callable[Q, R]]:
        """Register an implementation under a kind value.

        `@register()` uses the function name. `@register(key)` and
        `@register(name=key)` provide an explicit key.

        `Q` is its own type parameter, distinct from the dispatcher's `P`, because each
        kind routinely takes its own shape (a bound `functools.partial`, an impl that
        drops the fallback's catch-all `**kwargs`); only the return type `R` need agree.
        `bind` is where that shape is deliberately forgotten, since dispatch keys on a
        value at runtime rather than a type a checker can verify ahead of the call. When
        the key is a valid identifier that does not shadow the dispatcher's own API or
        state, the impl is also set as a dispatcher attribute, enabling direct
        `dispatcher.foo(...)` calls. Dispatch by key works either way.

        key: explicit dispatch key.
        name: explicit key, taking precedence over `key`.
        """
        dispatch_key = key if name is None else name

        def decorate(impl: Callable[Q, R]) -> Callable[Q, R]:
            return self.bind(
                impl,
                self.implied_key(impl) if dispatch_key is None else dispatch_key,
            )

        return decorate

    def implied_key[**Q](self, impl: Callable[Q, R]) -> str:
        """Derive the registry key from the impl's `__name__`, failing clearly when absent."""
        try:
            return impl.__name__
        except AttributeError:
            raise TypeError(
                f"cannot infer a kind for {impl!r} because it has no __name__. "
                f"Register it with an explicit key, for example register(name=...).",
            ) from None

    def bind[**Q](self, impl: Callable[Q, R], key: Hashable) -> Callable[Q, R]:
        """Store `impl` under `key`, also exposing it as an attribute when that is safe.

        The registry only ever calls a stored impl with the arguments a matching dispatch
        call actually supplies, never with `P` reconstructed from thin air, so re-typing
        `impl` here to the dispatcher's own `Callable[P, R]` costs no real safety, only the
        precision of `Q`, which this one boundary is exactly where that trade belongs.
        """
        if self.exposable(key):
            setattr(self, key, impl)
        self.registry_map[key] = cast("Callable[P, R]", impl)
        return impl

    def exposable(self, key: Hashable) -> TypeGuard[str]:
        """Whether `key` may become a dispatcher attribute without shadowing its own API.

        The key must be an identifier string colliding with neither class-level API
        (`register`, `kinds`, ...) nor instance state, except for the attribute that a
        previous registration of the same key already exposed.
        """
        if not (isinstance(key, str) and key.isidentifier()):
            return False
        if hasattr(type(self), key):
            return False
        return key not in vars(self) or vars(self)[key] is self.registry_map.get(key)

    @property
    def registry(self) -> MappingProxyType[Hashable, Callable[P, R]]:
        """Read-only view of the kind to impl mapping."""
        return MappingProxyType(self.registry_map)

    def kinds(self) -> list[Hashable]:
        """All registered kinds, sorted by their `repr` so mixed-type keys stay orderable."""
        return sorted(self.registry_map, key=repr)

    def __contains__(self, key: Hashable) -> bool:
        return key in self.registry_map

    def __getattr__(self, name: str) -> Callable[P, R]:
        """Type the attribute `bind` exposes for an identifier-shaped key (`dispatcher.foo`).

        Only reached when normal lookup misses, so a real attribute (`registry_map`, an
        exposed impl `bind` has actually set) is always found first; this exists solely to
        give the type checker a return type for the dynamic `dispatcher.foo(...)` surface
        `register` documents, which no static attribute list can otherwise describe.

        name: the exposed kind being looked up.
        """
        raise AttributeError(name)

    def __repr__(self) -> str:
        name = self.fallback.__name__ if self.fallback else "(unbound)"
        return f"<value_dispatch {name!r} kinds={self.kinds()}>"


class type_dispatch[**P, R]:
    """Turn a function into a type-dispatched generic, the dual of :class:`value_dispatch`.

    Where `value_dispatch` keys on the *value* of a keyword argument, this keys on the *type* of
    the first positional argument, the open type ladder a hand-rolled `if isinstance(x, A): ...
    elif isinstance(x, B): ...` chain encodes (rendering each GPU/NPU subtype, formatting each
    result kind). Like `functools.singledispatch` but registering with the same ergonomics as
    `value_dispatch`:

        @type_dispatch
        def render(node): ...               # the fallback for an unmatched type

        @render.register()
        def _(node: Circle): ...            # type read from the first parameter's annotation

        @render.register(Square)
        def _(node): ...                    # type given explicitly

    A call resolves the first argument's type against the registry by walking its MRO, so the most
    specific registered base wins and a subtype falls through to a registered supertype. A type the
    ladder does not cover routes to the originally decorated fallback. New cases are new
    registrations, never another `elif`.

    fallback: the wrapped catch-all function; supplied implicitly by the decorator.
    """

    def __init__(self, fallback: Callable[P, R] | None = None) -> None:
        self.registry_map: dict[type, Callable[P, R]] = {}
        self.fallback: Callable[P, R] | None = None
        if fallback is not None:
            self.bind_fallback(fallback)

    def bind_fallback(self, fallback: Callable[P, R]) -> None:
        """Adopt `fallback` as the catch-all and copy its metadata, rejecting a method fallback."""
        if inspect.isfunction(fallback) and fallback.__code__.co_varnames[:1] == ("self",):
            raise TypeError(
                "type_dispatch does not support methods. Decorate a module-level function "
                "instead of one whose first parameter is self.",
            )
        self.fallback = fallback
        functools.update_wrapper(self, fallback, updated=())

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Bind the fallback on the parametrised first call, else dispatch on `args[0]`'s type.

        With no fallback yet this is the bare-decorator stage and the sole argument is the function
        to wrap. Once bound, the first positional argument's type selects the implementation, and a
        call with no positional argument routes to the fallback.
        """
        if self.fallback is None:
            if len(args) != 1 or kwargs or not callable(args[0]):
                raise TypeError(
                    "this type_dispatch has no function bound yet. Apply it as a decorator to "
                    "its fallback function before calling it.",
                )
            self.bind_fallback(cast("Callable[P, R]", args[0]))
            return cast(R, self)
        impl = self.resolve(type(args[0]), self.fallback) if args else self.fallback
        return impl(*args, **kwargs)

    def resolve(self, cls: type, fallback: Callable[P, R]) -> Callable[P, R]:
        """The implementation for `cls`, the nearest registered base on its MRO, else `fallback`.

        cls: the runtime type of the first argument.
        fallback: returned when no base of `cls` is registered.
        """
        for base in cls.__mro__:
            if base in self.registry_map:
                return self.registry_map[base]
        return fallback

    def register[**Q](
        self,
        cls: type | None = None,
    ) -> Callable[[Callable[Q, R]], Callable[Q, R]]:
        """Register an implementation for a type.

        Two forms mirroring `value_dispatch`:
        - `@register()` reads the type from the first parameter's annotation.
        - `@register(SomeType)`: the type is given explicitly.

        `Q` is its own type parameter, distinct from the dispatcher's `P`: an impl's first
        parameter is annotated with the concrete type it handles, narrower than whatever
        the fallback declares, which is the whole point of registering it. `bind` is where
        that narrower shape is deliberately forgotten, the same trade `value_dispatch.bind`
        makes for the same reason.

        cls: explicit type, or `None` to infer it from the implementation.
        """
        return lambda impl: self.bind(
            self.annotated_type(impl) if cls is None else cls,
            impl,
        )

    def annotated_type[**Q](self, impl: Callable[Q, R]) -> type:
        """The type annotation of `impl`'s first parameter, failing clearly when it is absent."""
        hints = inspect.signature(impl).parameters
        first = next(iter(hints.values()), None)
        if first is None or first.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"cannot infer a dispatch type for {impl!r} because its first parameter has no "
                f"annotation. Register it with an explicit type, for example register(SomeType).",
            )
        if not isinstance(first.annotation, type):
            raise TypeError(
                f"the first parameter of {impl!r} is annotated {first.annotation!r}, which is not "
                f"a class. Register it with an explicit type instead.",
            )
        return first.annotation

    def bind[**Q](self, cls: type, impl: Callable[Q, R]) -> Callable[Q, R]:
        """Store `impl` as the implementation for `cls`, returning `impl` unchanged."""
        self.registry_map[cls] = cast("Callable[P, R]", impl)
        return impl

    @property
    def registry(self) -> MappingProxyType[type, Callable[P, R]]:
        """Read-only view of the type to impl mapping."""
        return MappingProxyType(self.registry_map)

    def types(self) -> list[type]:
        """All registered types, sorted by name so the listing is stable."""
        return sorted(self.registry_map, key=lambda cls: cls.__name__)

    def __contains__(self, cls: type) -> bool:
        return cls in self.registry_map

    def __repr__(self) -> str:
        name = self.fallback.__name__ if self.fallback else "(unbound)"
        types = [cls.__name__ for cls in self.types()]
        return f"<type_dispatch {name!r} types={types}>"
