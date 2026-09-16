# Changelog

All notable changes to patos are documented here.

The format follows Keep a Changelog, and releases are cut from the version in `pyproject.toml`.

## 0.0.14

### Added

- `SourceRegistration`, a frozen model that binds a research node to its governing
  source files and verifies the recorded SHA-256 seal. Paths are repository-relative,
  text uses LF-normalized bytes, and duplicate or missing sources are rejected.
- Portable text-digest helpers for individual files and ordered source sets.

### Changed

- Research-node seals cover statement sections while excluding front matter and
  settlement sections, allowing evidence and status updates without changing the
  registered claim.
- Release documentation uses uv, and the retired chefe pre-commit hook is removed.
- Ruff formatting excludes Markdown so it preserves MkDocs include directives and
  deliberate example alignment.

## 0.0.13

### Added

- `patos.torch`, behind a new `torch` extra, holding the run-wide RNG and precision controls
  (`seed_all`, `seeded`, `configure_torch`, `setup`) and the dtype-aware tensor helpers (`eps`,
  `tiny`, `eye_like`, `fp32_matmul_precision`). Torch is imported only inside that namespace, so
  the core stays pydantic-only and a consumer without the extra never pays for it.

  `seeded` is the one genuinely new name. The `fork_rng(devices=[])` plus `manual_seed` idiom it
  wraps had been copied nine times across one research tree, and every copy is a place where a
  reproducible draw can quietly start advancing the global stream that everything after it reads.

- `content_key`, the hex digest that keys a cached artifact by the content determining it. It
  lives beside `DerivedCache` as the on-disk counterpart to that in-memory key, and it is dtype
  free, so it stays in the core rather than in the torch extra.

- `Strategy.from_registry(root, name, **factory_kwargs)`, the bridge between the two halves of the
  pattern. `Registry` collects the concrete classes as they are imported and `Strategy` picks one
  by name at runtime, and until now every consumer joined them by hand with the same
  `for impl in Root.implementations()` loop, once per family. One codebase carried nine copies of
  it in a single constructor. Registrations stay lazy, so only the implementation actually
  selected is ever built.

### Changed

- `Shared` keeps one piece of slot state instead of two. A slot used to carry both a `built` flag
  and an optional resource, so the yielded handle read as `R | None` and needed a suppression to
  claim otherwise. The resource itself now says whether the slot is filled, which drops the
  suppression, the flag and the chance of the two disagreeing.

- CI bootstraps with the official `astral-sh/setup-uv` action and plain `uv run` steps rather than
  a shared action hosted in another repository. A package that ships to PyPI has to be buildable
  from its own checkout, and a shared bootstrap made every release depend on somebody else's
  default branch.

## 0.0.12

### Added

- `OpenModel` and `FrozenOpenModel`, the bases for a payload somebody else authors. They declare
  the fields the reader consumes and drop whatever the provider advertises beside them, which is
  what OIDC discovery documents, OpenAI-compatible APIs and most REST providers do by design and
  by spec. The strict bases stay strict, because a payload we author carries exactly what it
  declares and a stray key there is a typo worth failing on.

  This closes the gap 0.0.10 opened. Forbidding extras is right for our own payloads and wrong
  for somebody else's, and with only strict bases available the second case had to be spelled as
  a per-model `ConfigDict(extra="ignore")` that every future integration model would have to
  remember. The one that forgot took a deployment down at boot when its identity provider began
  advertising token introspection and back-channel logout metadata. The choice now lives in the
  base class, where it is visible at the class statement and impossible to forget silently.

  Unknown fields are dropped rather than kept, so a parsed record still carries exactly its
  declared fields, nothing downstream can come to depend on a key the provider never promised,
  and `stable_id` stays a function of the declared schema instead of shifting whenever an
  upstream service adds metadata.

## 0.0.11

### Fixed

- The core package requires Python 3.13 again rather than 3.14. Only the `sql` extra needs 3.14,
  because `sql/templates.py` imports `string.templatelib` and `PK(UUID7)` calls `uuid.uuid7`, and
  an extra cannot carry a floor of its own. The 3.14 requirement published in 0.0.10 excluded
  every consumer stuck on 3.13, which is a real environment rather than a stale one, since sglang
  publishes no CPython 3.14 wheel in any release up to 0.5.16 and a GPU serving environment built
  on it imports `Strategy`, `Registry` and `SingletonMeta` from here. Install `patos[sql]` on 3.14.

## 0.0.10

### Added

- `Runtime[T]`, a field annotation that treats one already-validated live object as an opaque
  runtime value, so a `FrozenModel` can carry callables, locks, syntax trees, tensors, or clients
  in selected fields without opening the whole model to arbitrary types.
- `FrozenModel.stable_id`, a deterministic cached 64-bit identity derived from the model's
  qualified name and its validated fields, so equal values hash equal across processes.
- The `sql` extra now also installs `sqlmodel` and `typing-extensions`, which the typed SQL
  namespace imports.

### Changed

- `Model` and `FrozenModel` now forbid extra fields, matching the house contract that a model
  carries exactly what it declares.
- `FrozenFlexModel` extends `FrozenModel`, so it inherits `stable_id`, the forbidden extras, and
  the alias population alongside arbitrary type support.
- `available` is renamed `is_available` so the probe reads as a question at every call site.

### Fixed

- `ty` no longer points at a monorepo-local interpreter path, so the check runs in any checkout
  with the project's own environment.

## 0.0.9

### Added

- The optional `patos[sql]` extra with the `patos.sql` namespace. It provides typed SQLModel
  columns, native PostgreSQL enums, JSONB access, pgvector cosine distance, typed `VALUES`
  relations, PostgreSQL digest, hexadecimal and UUIDv8 hashing, and SQL template helpers without
  adding those dependencies to the core package. Hash algorithms are OpenSSL names resolved by
  the database rather than a hardcoded Python allowlist.
- `PGEnum.name`, `PGEnum.values`, and `PGEnum.type` derive native PostgreSQL enum metadata from
  the qualified Python class. Nested names such as `Watermark.Kind` map to
  `watermark_kind`, and enum values are persisted exactly as declared.

### Changed

- Reusable PostgreSQL helpers moved out of AIZK into Patos so database projects share one typed
  interface through `from patos import sql`.

## 0.0.8

### Added

- `Registry.select(predicate)` returns the concrete implementations satisfying a predicate, and `Registry.first_available(probe)` returns the first whose availability probe passes, so consumers stop hand-rolling the "iterate the registry and pick" filter. The module-level `available` helper is the default probe.
- `type_dispatch`, the dual of `value_dispatch`, dispatches on the type of the first positional argument by walking its MRO (most specific wins), the open-type-ladder replacement for an `isinstance` chain.
- `Decorator`, a transparent delegation base that forwards every non-overridden attribute to a wrapped object, so a wrapper restates only what it changes.
- `Pipeline` and the `Reversible` stage protocol, a reversible stack applied forward and unwound in reverse around a core operation.
- `Lifecycle` and `IllegalTransition`, a typed state machine that permits only the transitions its table declares and raises on any other.
- `DerivedCache`, a load-once cache of a derived value keyed by exactly the fields it depends on.

### Fixed

- `value_dispatch.register` and `type_dispatch.register` now always return a decorator. The inferred forms use `@register()`, which removes the callable-key ambiguity and preserves implementation signatures under MyPy and Pyrefly.
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
