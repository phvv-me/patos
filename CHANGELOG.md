# Changelog

All notable changes to patos are documented here.

The format follows Keep a Changelog, and releases are cut from the version in `pyproject.toml`.

## 0.0.4

### Added

- `Registry.implementations()` lists the concrete enrolled classes, dropping the registry root itself and any abstract bases. It replaces the hand-rolled `for c in Base.registry() if c is not Base and not c.__abstractmethods__` filter that real consumers (GPU/NPU/Tracer providers, the gauge family) kept rewriting.
- `Registry.find(name, attr="name")` looks an implementation up by a keyed class attribute, raising a clear `KeyError` that lists the known keys. It replaces the `{c.name: c for c in Base.registry()}[name]` dict that keyed registries hand-rolled.
- `Registry.root()` returns the registry root that owns a class's implementation list.

### Changed

- `Registry.registry()` is now typed `list[type[Self]]`, so `Base.registry()` and `Base.dispatch()` carry the precise element type instead of `type[Registry]`, dropping the `cast` consumers needed.
- `Registry.dispatch` now walks `implementations()`, so abstract intermediate bases are skipped instead of being tried and failing.

## 0.0.3

### Added

- Export `SingletonMeta` so a singleton can be declared `class X(metaclass=SingletonMeta)`, matching `FlyweightMeta` for metaclass-first usage.

## 0.0.2

### Changed

- Flyweight is now metaclass-only. Use `class X(metaclass=FlyweightMeta)`; the `Flyweight` base class is gone.
- The package surface is eager and explicit, no lazy `__getattr__`, so `from patos import StrFlag` and friends resolve directly.

### Removed

- Removed the `flags` helper. It built CLI argv from kwargs, which is a utility rather than a design pattern.
- Removed the internal catalog module and the copy-in CLI. The documentation is the copy-in path now.

### Docs

- New duck logo, raster banner, and an es-toolkit-style docs site. English only.

## 0.0.1

### Added

- Initial release with seven patos. registry, singleton, flyweight, strategy, dispatch, flags, strflag.
- Two ways to use, install from PyPI or copy a single self-contained module from its docs page.
- Python 3.11 through 3.14, fully typed, zero runtime dependencies.
