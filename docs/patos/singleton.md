# singleton

One instance per class, built at most once, with `__init__` that runs exactly one time.

## When to reach for it

Reach for `singleton` when a class represents something there should be only one of, like a
shared config, a connection pool, or a process-wide cache. Every construction returns the same
object, and each class keeps its own instance, so subclasses are independent singletons.

## The anti-pattern it replaces

It replaces the cached-`__new__` trick, which returns the same object but re-runs `__init__` on
every call and quietly re-initializes your shared state. By owning `__call__` on the metaclass,
`singleton` returns the cached instance without re-initializing it. It is the no-argument
degenerate case of the flyweight pattern.

## Usage

```python
from patos import Singleton


class Settings(Singleton):
    def __init__(self) -> None:
        self.debug = False


a = Settings()
a.debug = True
b = Settings()      # same object, __init__ did not run again

assert a is b
assert b.debug is True
```

## Public API

- `Singleton`. Base class. Subclass it and every construction returns the shared instance.
- `SingletonMeta`. The metaclass that owns `__call__` and caches one instance per class. Use it directly when you already have a metaclass to compose with.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/singleton.py"
```
