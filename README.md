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

## Patterns

### Creational

| pato | description | docs |
|---|---|---|
| `Singleton` | one instance per class, `__init__` runs once | [docs](https://phvv.me/patos/patos/singleton/) |
| `FlyweightMeta` | a metaclass that interns instances by constructor args, build-once and immutable | [docs](https://phvv.me/patos/patos/flyweight/) |

### Dispatch & selection

| pato | description | docs |
|---|---|---|
| `Registry` | self-registering class hierarchies via `__init_subclass__`, with try-each `dispatch` | [docs](https://phvv.me/patos/patos/registry/) |
| `Strategy` | a named family of interchangeable implementations, with `first_available()` | [docs](https://phvv.me/patos/patos/strategy/) |
| `value_dispatch` | dispatch on a value the way `singledispatch` dispatches on a type | [docs](https://phvv.me/patos/patos/dispatch/) |

### Command-line

| pato | description | docs |
|---|---|---|
| `StrFlag` | an enum `Flag` whose members carry a literal string, OR-combinable and iterable | [docs](https://phvv.me/patos/patos/strflag/) |

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

This project uses [uv](https://docs.astral.sh/uv/).

- Install: `uv sync --extra dev`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- Typecheck: `uv run pyrefly check`
- Test: `uv run pytest`
- Docs: `uv run --extra docs mkdocs build`
- Build: `uv build`
