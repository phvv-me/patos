# API

The stable public surface of patos. Every name here is importable straight from the top-level
package.

```python
from patos import (
    Singleton,
    FlyweightMeta,
    Registry,
    Strategy,
    value_dispatch,
    StrFlag,
)
```

## Two ways to use a pato

Install the flock and import what you want.

```sh
pip install patos
```

```python
from patos import Registry, Singleton
```

Or copy a single duck. Open its page, copy the source from the **Source** section, and paste it
into your own project. No dependency, no version pin, no tool, just one module you own.

## The flock

### Creational

| pato | exports | what it does |
|---|---|---|
| [singleton](patos/singleton.md) | `Singleton` | one instance per class, `__init__` runs once |
| [flyweight](patos/flyweight.md) | `FlyweightMeta` | intern instances by constructor args, build-once and immutable |

### Dispatch & selection

| pato | exports | what it does |
|---|---|---|
| [registry](patos/registry.md) | `Registry` | self-registering class hierarchies via `__init_subclass__`, with try-each `dispatch` |
| [strategy](patos/strategy.md) | `Strategy` | a named family of interchangeable implementations, with `first_available()` |
| [dispatch](patos/dispatch.md) | `value_dispatch` | dispatch on a value the way `singledispatch` dispatches on a type |

### Command-line

| pato | exports | what it does |
|---|---|---|
| [strflag](patos/strflag.md) | `StrFlag` | an enum `Flag` whose members carry a literal string, OR-combinable and iterable |
