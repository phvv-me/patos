from __future__ import annotations

from typing import Generic, TypeVar

W = TypeVar("W")


class Decorator(Generic[W]):
    """A transparent wrapper forwarding everything to its `wrapped` object but what it overrides.

    The lifted delegation base the codec combinators hand-roll: each wraps one inner object and
    re-implements only the few methods whose behaviour it changes (a rate adjustment, a pre/post
    transform, an outlier carve), wanting every other attribute to read straight through. Rather
    than each decorator writing its own `__getattr__` (or restating a long protocol's worth of
    pass-through methods), subclass this and override only what differs, the rest forwards to
    `wrapped` automatically.

    Attribute *reads* missing on the decorator fall through to `wrapped`, so its methods,
    properties and data are all visible unless shadowed. Writes and deletes land on the decorator
    itself, never mutating the wrapped object behind its back. It is generic over the wrapped type,
    so `Decorator[Quantizer]` keeps the forwarded surface typed as a `Quantizer`.

    wrapped: the object every non-overridden access delegates to.
    """

    def __init__(self, wrapped: W) -> None:
        self.wrapped = wrapped

    def __getattr__(self, name: str) -> object:
        """Forward an attribute missing on the decorator to the wrapped object.

        `__getattr__` runs only after normal lookup misses, so an overriding method or a field set
        on the decorator (including `wrapped` itself) is found first and never re-routed. During
        construction `wrapped` is not yet set, so a delegated access before then raises a clear
        `AttributeError` naming the attribute rather than recursing.

        name: the attribute being looked up.
        """
        try:
            wrapped = object.__getattribute__(self, "wrapped")
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r} and no wrapped object yet.",
            ) from None
        return getattr(wrapped, name)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} wrapping {self.wrapped!r}>"
