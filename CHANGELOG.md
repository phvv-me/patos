# Changelog

All notable changes to patos are documented here.

The format follows Keep a Changelog, and releases are cut from the version in `pyproject.toml`.

## 0.0.8

### Added

- `Registry.select(predicate)` returns the concrete implementations satisfying a predicate, and `Registry.first_available(probe)` returns the first whose availability probe passes, so consumers stop hand-rolling the "iterate the registry and pick" filter. The module-level `available` helper is the default probe.
- `type_dispatch`, the dual of `value_dispatch`, dispatches on the type of the first positional argument by walking its MRO (most specific wins), the open-type-ladder replacement for an `isinstance` chain.
- `Decorator`, a transparent delegation base that forwards every non-overridden attribute to a wrapped object, so a wrapper restates only what it changes.
- `Pipeline` and the `Reversible` stage protocol, a reversible stack applied forward and unwound in reverse around a core operation.
- `Lifecycle` and `IllegalTransition`, a typed state machine that permits only the transitions its table declares and raises on any other.
- `DerivedCache`, a load-once cache of a derived value keyed by exactly the fields it depends on.

### Fixed

- `Registry` auto-naming now splits embedded acronyms, so `HTTPServer` derives to `http-server` and `XMLHttpRequest` to `xml-http-request` instead of fusing the acronym into the next word.
- `Registry` auto-naming keeps a pure acronym whole even when it carries a digit, so the real codec `E8P` derives to `e8p` (not the broken `e8-p`) while a capital that begins a new word still splits (`E8Lattice` to `e8-lattice`). The derived key is now idempotent, which is what makes the `find` round-trip stable.
- A bare `name: str` annotation on a subclass no longer suppresses kebab derivation. Earlier it skipped derivation without assigning anything, so the subclass silently inherited the root's key and answered `find` for the wrong name.

## 0.0.6

### Changed

- `StrategyError` subclasses `LookupError` so its message renders verbatim, and it is exported at the package top level along with `Available`.
- `Registry.dispatch` raises an `ExceptionGroup` carrying every implementation's refusal instead of only the last error with no chaining.
- `Registry.find` matches own attributes only, so an inherited `name` no longer masquerades as a registration, and duplicate keys raise instead of silently last winning.
- `SingletonMeta` stores the instance on the class itself, mirroring the flyweight, so classes are no longer pinned for the process lifetime by a global registry.
- `FlyweightMeta` interns by argument types as well as values, so `Node(1)`, `Node(True)`, and `Node(1.0)` stay distinct.

### Fixed

- `value_dispatch.register` treated a callable dispatch key (a class, a partial) as the implementation, and `bind` could clobber the dispatcher's own API when an implementation was named `register` or `fallback`.
- A parametrised dispatcher called before binding a function raises a clear `TypeError` instead of a bare `IndexError`, and a method style fallback whose first parameter is `self` is rejected with guidance.
- `Strategy.factory` invalidates the resolution cache, so re registering a factory takes effect.
- `first_available` accepts a plain boolean `available` attribute instead of crashing on a non callable.
- Composite and empty `StrFlag` members now carry `.string`, joining their decomposed members' literals.
- A class inheriting from two registry roots enrolls in both registries.

## 0.0.5

### Changed

- Typing is now mypy strict with `disallow_any_explicit`, and mypy runs in CI. The pattern primitives use `ParamSpec` and `TypeVar` generics instead of `Any`, with `object` only on the genuinely variadic metaclass and registry forwarders.
- The docs adopt the shared Open Props design language over mkdocs-material, with a legible app-icon as logo and favicon, and a working `llms.txt` from the english post-build hook.
- CI actions updated to setup-uv v7, upload-pages-artifact v5, deploy-pages v5, and gh-release v3.

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
