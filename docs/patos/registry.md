# registry

Self-registering class hierarchies, where every subclass enrolls itself the moment it is defined.

## When to reach for it

Reach for `registry` when you have a base class and a growing set of implementations, and you
want new implementations to become available just by being imported. No central list to edit,
no manual registration call. Each direct child of `Registry` becomes a root that owns the list
of its own concrete subclasses, and `dispatch` walks that list trying each one until a build
succeeds.

## The anti-pattern it replaces

It replaces the hand-maintained dictionary that maps a name or a type to a class, the one that
every new implementation has to remember to update. It also replaces the `if isinstance(...)`
ladder that grows a branch per case. Registration moves into `__init_subclass__`, so adding a
case is adding a class.

## Usage

```python
from patos import Registry


class Loader(Registry):
    """Root of the loader flock. Each subclass tries to handle a path."""

    def __init__(self, path: str) -> None:
        self.path = path


class JsonLoader(Loader):
    @classmethod
    def from_dispatch(cls, path: str) -> "JsonLoader":
        if not path.endswith(".json"):
            raise ValueError("not a json path")
        return cls(path)


class YamlLoader(Loader):
    @classmethod
    def from_dispatch(cls, path: str) -> "YamlLoader":
        if not path.endswith((".yml", ".yaml")):
            raise ValueError("not a yaml path")
        return cls(path)


Loader.registry()           # [JsonLoader, YamlLoader]
Loader.dispatch("a.yaml")   # YamlLoader instance, first impl that accepts the path
```

## Public API

- `Registry`. Base mixin. Subclass it to start a flock.
- `Registry.registry()`. List the concrete implementations owned by this class's nearest root.
- `Registry.is_registry_root()`. Whether this class owns a registry.
- `Registry.dispatch(*args, **kwargs)`. Try each implementation's `from_dispatch` and return the first success, re-raising the last error if all fail.
- `Registry.from_dispatch(*args, **kwargs)`. Classmethod each implementation overrides to accept or reject the arguments.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/registry.py"
```
