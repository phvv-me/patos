# Changelog

All notable changes to patos are documented here.

The format follows Keep a Changelog, and releases are cut from the version in `pyproject.toml`.

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
