from __future__ import annotations

import functools
from collections.abc import Callable, Hashable
from types import MappingProxyType
from typing import Any


class value_dispatch:
    """Turn a function into a value-dispatched generic.

    Like `functools.singledispatch`, but the dispatch key is the *value* of a
    keyword argument (default `"kind"`) rather than the *type* of the first
    positional argument. Both decorator forms work:

        @value_dispatch
        def render(node, **kw): ...        # bare: kind is "kind"

        @value_dispatch(kind="how")
        def render(node, **kw): ...        # kind is "how"

    Implementations register under a key with `@render.register`. Each is also
    exposed as an attribute (`render.html`) so callers that already know the
    kind can skip the registry lookup. Calling with no kind (or `kind=None`)
    falls back to the originally decorated function.

    fallback: the wrapped function; supplied implicitly by the decorator.
    kind: name of the keyword argument carrying the dispatch value.
    """

    def __init__(
        self,
        fallback: Callable[..., Any] | None = None,
        *,
        kind: str = "kind",
    ) -> None:
        self.kind_arg = kind
        self.registry_map: dict[Hashable, Callable[..., Any]] = {}
        self.fallback: Callable[..., Any] | None = None
        if fallback is not None:
            self.bind_fallback(fallback)

    def bind_fallback(self, fallback: Callable[..., Any]) -> None:
        """Adopt `fallback` as the default impl and copy its metadata onto self."""
        self.fallback = fallback
        functools.update_wrapper(self, fallback)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Bind the fallback on the parametrised first call, otherwise dispatch.

        With no fallback yet, this is the parametrised-decorator stage: the sole
        argument is the function to wrap. Once bound, pop the kind keyword and
        route to its registered impl, falling back when the kind is absent.
        """
        if self.fallback is None:
            self.bind_fallback(args[0])
            return self
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

    def register(
        self,
        arg: Hashable | Callable[..., Any] | None = None,
        *,
        name: Hashable | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register an implementation under a kind value.

        Three forms:
        - `@register` bare: key is the function's `__name__`.
        - `@register("foo")` or `@register(name="foo")`: explicit key.
        - `register(existing, name="alias")`: register an existing function.

        When the key is a valid identifier the impl is also set as a dispatcher
        attribute, enabling direct `dispatcher.foo(...)` calls.

        arg: explicit key, or the function itself in the bare/direct forms.
        name: explicit key, taking precedence over `arg` and the `__name__`.
        """
        if callable(arg):
            return self.bind(arg, arg.__name__ if name is None else name)
        key = arg if name is None else name

        def decorate(impl: Callable[..., Any]) -> Callable[..., Any]:
            return self.bind(impl, impl.__name__ if key is None else key)

        return decorate

    def bind(self, impl: Callable[..., Any], key: Hashable) -> Callable[..., Any]:
        """Store `impl` under `key`, also exposing it as an attribute when key is an identifier."""
        self.registry_map[key] = impl
        if isinstance(key, str) and key.isidentifier():
            setattr(self, key, impl)
        return impl

    @property
    def registry(self) -> MappingProxyType[Hashable, Callable[..., Any]]:
        """Read-only view of the kind to impl mapping."""
        return MappingProxyType(self.registry_map)

    def kinds(self) -> list[Hashable]:
        """All registered kinds, sorted by their `repr` so mixed-type keys stay orderable."""
        return sorted(self.registry_map, key=repr)

    def __contains__(self, key: Hashable) -> bool:
        return key in self.registry_map

    def __repr__(self) -> str:
        name = self.fallback.__name__ if self.fallback else "(unbound)"
        return f"<value_dispatch {name!r} kinds={self.kinds()}>"
