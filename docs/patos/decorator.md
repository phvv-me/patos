# decorator

A transparent wrapper that forwards everything to the object it wraps, except the few things a subclass chooses to override.

## When to reach for it

Reach for `Decorator` when you want to wrap one object, change a little of its behaviour, and
leave everything else reading straight through. A subclass overrides only the methods it
actually changes, and every other attribute (methods, properties, data) is delegated to the
wrapped object automatically.

## The anti-pattern it replaces

It replaces the per-wrapper `__getattr__` each decorator writes by hand, and the long list of
pass-through methods a wrapper restates just to forward a protocol it does not change. Both are
boilerplate that drifts out of sync with the wrapped type. Subclass `Decorator`, override what
differs, and the rest forwards itself.

## Usage

```python
from patos import Decorator


class Codec:
    def encode(self, x): ...
    def decode(self, codes): ...
    def bits(self) -> int:
        return 8


class Doubled(Decorator[Codec]):
    """Same codec, but it reports twice the rate."""

    def bits(self) -> int:
        return self.wrapped.bits() * 2


inner = Codec()
loud = Doubled(inner)
loud.bits()        # 16 -- overridden
loud.encode(...)   # delegated to inner.encode
```

Writes land on the decorator, never on the wrapped object:

```python
loud.tag = "x"     # sets tag on the decorator, leaves inner untouched
```

## Public API

- `Decorator(wrapped)`. Wrap an object. Every non-overridden attribute access delegates to it.
- `Decorator.wrapped`. The wrapped object the decorator forwards to.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/decorator.py"
```
