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
    type_dispatch,
    Decorator,
    Pipeline,
    Lifecycle,
    DerivedCache,
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

The SQL namespace is an optional extension rather than a top level import. Install its extra and
import the namespace when a project uses PostgreSQL.

```sh
pip install "patos[sql]"
```

```python
from patos import sql
```

## The flock

### Creational

| pato | exports | what it does |
|---|---|---|
| [singleton](patos/singleton.md) | `Singleton` | one instance per class, `__init__` runs once |
| [flyweight](patos/flyweight.md) | `FlyweightMeta` | intern instances by constructor args, build-once and immutable |

### Dispatch & selection

| pato | exports | what it does |
|---|---|---|
| [registry](patos/registry.md) | `Registry` | self-registering class hierarchies via `__init_subclass__`, with `dispatch`, `select` and `first_available` |
| [strategy](patos/strategy.md) | `Strategy` | a named family of interchangeable implementations, with `first_available()` |
| [dispatch](patos/dispatch.md) | `value_dispatch`, `type_dispatch` | dispatch on a value, or on the first argument's type the way `singledispatch` does |

### Structural & lifecycle

| pato | exports | what it does |
|---|---|---|
| [decorator](patos/decorator.md) | `Decorator` | a transparent wrapper forwarding everything to the wrapped object but its overrides |
| [pipeline](patos/pipeline.md) | `Pipeline`, `Reversible` | a reversible stack of stages, applied forward and unwound in reverse |
| [lifecycle](patos/lifecycle.md) | `Lifecycle`, `IllegalTransition` | a typed state machine that permits only the transitions it declares |
| [cache](patos/cache.md) | `DerivedCache` | a load-once cache of a derived value keyed by exactly what it depends on |

### Command-line

| pato | exports | what it does |
|---|---|---|
| [strflag](patos/strflag.md) | `StrFlag` | an enum `Flag` whose members carry a literal string, OR-combinable and iterable |

### Research registration

`SourceRegistration(repository=..., node=..., sources=...)` exposes `digest()` and
`verify()`. The node must belong to the explicit source set. Verification checks its
single `registration_sha256` frontmatter field. The node's statement ends before
Evidence, Ledger, or Log; all other files contribute their full LF-normalized bytes.
Missing, duplicate, and outside-repository paths fail. Use this shared implementation
instead of copying registration hashing into an experiment.

### Optional extensions

| extension | exports | what it does |
|---|---|---|
| [sql](patos/sql.md) | `sql.Column`, `sql.PGEnum`, `sql.TypedJSONB`, `sql.CosineHalfvec`, SQL expression helpers | concise typed PostgreSQL and SQLModel building blocks installed through `patos[sql]` |
