# API

The stable public surface of patos. Every name here is importable straight from the top-level
package, which lazily loads the one pato module that defines it.

```python
from patos import Registry, Singleton, Flyweight, FlyweightMeta, Strategy, value_dispatch, flags, StrFlag
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

| pato | exports | what it does |
|---|---|---|
| [registry](patos/registry.md) | `Registry` | self-registering class hierarchies via `__init_subclass__` |
| [singleton](patos/singleton.md) | `Singleton` | one instance per class, `__init__` runs once |
| [flyweight](patos/flyweight.md) | `Flyweight`, `FlyweightMeta` | intern instances by constructor args, build-once and immutable |
| [strategy](patos/strategy.md) | `Strategy` | a named family of interchangeable implementations |
| [dispatch](patos/dispatch.md) | `value_dispatch` | dispatch on a value the way `singledispatch` dispatches on a type |
| [flags](patos/flags.md) | `flags` | turn keyword options into a CLI argv tuple |
| [strflag](patos/strflag.md) | `StrFlag` | an enum `Flag` whose members carry a literal string, OR-combinable and iterable |
