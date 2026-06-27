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


Loader.registry()           # [Loader, JsonLoader, YamlLoader]
Loader.implementations()    # [JsonLoader, YamlLoader] -- concrete only, root dropped
Loader.dispatch("a.yaml")   # YamlLoader instance, first impl that accepts the path
```

When implementations carry a key, look one up by name instead of walking the list:

```python
class Loader(Registry):
    name = "base"


class JsonLoader(Loader):
    name = "json"


Loader.find("json")         # JsonLoader
Loader.find("toml")         # KeyError listing the known names
```

## Public API

- `Registry`. Base mixin. Subclass it to start a flock.
- `Registry.registry()`. List every enrolled class, the root included, owned by this class's nearest root.
- `Registry.implementations()`. The concrete enrolled classes, dropping the root itself and any abstract bases. The view to fan out over real providers.
- `Registry.find(name, attr="name")`. Return the implementation whose own `attr` equals `name`, raising a clear `KeyError` listing the known keys when missing and a `ValueError` on duplicate keys.
- `Registry.select(predicate)`. The concrete implementations satisfying `predicate`, in registration order, replacing the hand-rolled `[impl for impl in implementations() if ...]` filter.
- `Registry.first_available(probe=available)`. The first implementation whose availability `probe` passes, raising a `LookupError` on none. The default probe reads an `available()` / `is_available()` method, counting a class without one as always available.
- `Registry.root()`. The registry root that owns this class's implementation list.
- `Registry.is_registry_root()`. Whether this class owns a registry.
- `Registry.dispatch(*args, **kwargs)`. Try each implementation's `from_dispatch` and return the first success, raising every refusal together as an `ExceptionGroup` if all fail.
- `Registry.from_dispatch(*args, **kwargs)`. Classmethod each implementation overrides to accept or reject the arguments.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/registry.py"
```
