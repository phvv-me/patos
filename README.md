<p align="center">
  <img src="https://raw.githubusercontent.com/phvv-me/patos/main/docs/assets/banner.png" alt="patos" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/patos/"><img src="https://img.shields.io/pypi/v/patos?color=2563EB&label=pypi" alt="PyPI version"></a>
  <a href="https://pypi.org/project/patos/"><img src="https://img.shields.io/pypi/pyversions/patos?color=2563EB" alt="Python versions"></a>
  <a href="https://github.com/phvv-me/patos/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2563EB.svg" alt="License"></a>
  <a href="https://phvv.me/patos"><img src="https://img.shields.io/badge/docs-phvv.me%2Fpatos-2563EB" alt="Docs"></a>
</p>

A flock of typed, zero-dependency Python design patterns. Each pattern is a single self-contained `pato` (a duck you adopt), fully typed, standard library only, and small enough to read in one sitting.

## Install

```sh
pip install patos
```

Or copy a single duck. Open its docs page, copy the source from the Source section, and paste it into your own project. No dependency, no version pin, no tool, just one module you own.

The optional PostgreSQL toolkit installs SQLAlchemy, pgvector, and inflection only when a
project asks for them.

```sh
pip install "patos[sql]"
```

```python
from patos import sql
```

## Patterns

### Creational

| pato | description | docs |
|---|---|---|
| `Singleton` | one instance per class, `__init__` runs once | [docs](https://phvv.me/patos/patos/singleton/) |
| `FlyweightMeta` | a metaclass that interns instances by constructor args, build-once and immutable | [docs](https://phvv.me/patos/patos/flyweight/) |

### Dispatch & selection

| pato | description | docs |
|---|---|---|
| `Registry` | self-registering class hierarchies via `__init_subclass__`, with try-each `dispatch`, `select(predicate)` and `first_available()` | [docs](https://phvv.me/patos/patos/registry/) |
| `Strategy` | a named family of interchangeable implementations, with `first_available()` | [docs](https://phvv.me/patos/patos/strategy/) |
| `value_dispatch` | dispatch on a value the way `singledispatch` dispatches on a type | [docs](https://phvv.me/patos/patos/dispatch/) |
| `type_dispatch` | dispatch on the first argument's type, the open-type-ladder dual of `value_dispatch` | [docs](https://phvv.me/patos/patos/dispatch/) |

### Structural & lifecycle

| pato | description | docs |
|---|---|---|
| `Decorator` | a transparent wrapper that forwards everything to the wrapped object but what it overrides | [docs](https://phvv.me/patos/patos/decorator/) |
| `Pipeline` | a reversible stack of stages, applied forward and unwound in reverse around a core op | [docs](https://phvv.me/patos/patos/pipeline/) |
| `Lifecycle` | a typed state machine that permits only the transitions its table declares | [docs](https://phvv.me/patos/patos/lifecycle/) |
| `DerivedCache` | a load-once cache of a derived value keyed by exactly what it depends on | [docs](https://phvv.me/patos/patos/cache/) |

### Command-line

| pato | description | docs |
|---|---|---|
| `StrFlag` | an enum `Flag` whose members carry a literal string, OR-combinable and iterable | [docs](https://phvv.me/patos/patos/strflag/) |

### Optional extensions

| namespace | description | docs |
|---|---|---|
| `sql` | typed SQLModel columns, native PostgreSQL enums, pgvector cosine distance, JSONB reads, and set based SQL helpers | [docs](https://phvv.me/patos/patos/sql/) |

## Example

```python
from patos import Singleton


class Settings(Singleton):
    def __init__(self) -> None:
        self.debug = False


a = Settings()
a.debug = True
b = Settings()      # same object, __init__ did not run again

assert a is b and b.debug
```

## Documentation

Full documentation lives at [phvv.me/patos](https://phvv.me/patos).

## Development

- Install: `pip install -e ".[dev]"`
- Lint: `ruff check . && ruff format --check .`
- Typecheck: `pyrefly check`
- Test: `pytest`
- Docs: `pip install -e ".[docs]" && mkdocs build`
- Build: `python -m build`
