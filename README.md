# patos

[![CI](https://github.com/phvv-me/patos/actions/workflows/ci.yml/badge.svg)](https://github.com/phvv-me/patos/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-phvv.me%2Fpatos-2563EB)](https://phvv.me/patos)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[🇧🇷](https://phvv.me/patos/pt-BR/) [🇲🇽](https://phvv.me/patos/es/) [🇯🇵](https://phvv.me/patos/ja/) [🇨🇳](https://phvv.me/patos/zh/)

A flock of typed Python patterns. Pip one in, or copy one in and own the file.

Each pattern is a single self-contained `pato`, a duck you adopt. Every duck is fully typed,
standard library only, and small enough to read in one sitting.

## Two ways to use a pato

Install the whole flock and import the patterns you want.

```sh
pip install patos
```

```python
from patos import Registry
```

Or copy a single duck. Open its docs page, copy the source from the Source section, and paste it
into your own project. No dependency, no version pin, no tool, just one module you own.

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

## The flock

| pato | does |
|---|---|
| `registry` | self-registering class hierarchies via `__init_subclass__` |
| `singleton` | one instance per class, `__init__` runs once |
| `flyweight` | intern instances by constructor args, build-once and immutable |
| `strategy` | a named family of interchangeable implementations |
| `dispatch` | dispatch on a value the way `singledispatch` dispatches on a type |
| `flags` | turn keyword options into a CLI argv tuple |
| `strflag` | an enum `Flag` whose members carry a literal string, OR-combinable and iterable |

## Documentation

Full documentation lives at [https://phvv.me/patos](https://phvv.me/patos).

## Development

This project's dev environment is managed by [chefe](https://phvv.me/chefe).

- Install: `chefe install`
- Lint: `chefe run lint`
- Typecheck: `chefe run typecheck`
- Test: `chefe run test`
- Docs: `chefe run docs`
- Build: `uv build`
