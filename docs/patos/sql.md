# SQL

`patos.sql` is the optional typed PostgreSQL toolkit. It keeps SQLAlchemy, SQLModel, pgvector,
and inflection out of the core Patos installation.

## Install

```sh
pip install "patos[sql]"
```

Import one namespace and reach every public helper through it.

```python
from patos import sql
```

## Typed model columns

`sql.Column[T]` behaves as `T` on model instances and as a typed SQL expression on model
classes. It erases to the plain annotation at runtime so SQLModel and Pydantic see the value
type they expect.

```python
from datetime import datetime
from typing import cast

from sqlmodel import Field, SQLModel, select

from patos import sql


class Note(SQLModel, table=True):
    id: sql.Column[int] = Field(primary_key=True)
    title: sql.Column[str]
    attributes: sql.Column[dict] = Field(default_factory=dict, sa_type=sql.TypedJSONB)
    embedding: sql.Column[list[float] | None] = Field(
        default=None,
        sa_type=cast(type[list[float]], sql.CosineHalfvec(768)),
    )
    created_at: sql.Column[datetime]


statement = select(Note).where((Note.embedding @ ([1.0] * 768)) < 0.3)
```

`sql.Expr[T]` names the corresponding typed class expression for APIs that accept SQL columns.

## Native PostgreSQL enums

Subclass `sql.PGEnum` and use `.type` directly in a SQLAlchemy column. The PostgreSQL type name
comes from the qualified Python class name. Values are stored exactly as declared.

```python
from enum import auto

from patos import sql
from sqlalchemy import Column


class Watermark:
    class Kind(sql.PGEnum):
        graph = auto()
        profile = auto()


kind = Column(Watermark.Kind.type, nullable=False)

assert Watermark.Kind.name == "watermark_kind"
assert Watermark.Kind.values == ("graph", "profile")
```

## JSONB and vector operators

`sql.TypedJSONB` makes common JSON reads typed. `>>` reads text, while indexing with a Python
type casts the JSON text in PostgreSQL.

```python
title = Note.attributes >> "title"
count = Note.attributes[int] >> "count"
distance = Note.embedding @ query_vector
```

`sql.CosineHalfvec` maps `@` to pgvector cosine distance. This keeps distance work in
PostgreSQL and removes repeated operator boilerplate from application code.

## Set based helpers

- `sql.relation` builds a named typed `VALUES` relation for bulk joins and writes.
- `sql.digest` computes a binary digest with PostgreSQL `pgcrypto`.
- `sql.hex` encodes that digest as lowercase hexadecimal text.
- `sql.uuid8` stores the first 128 digest bits as an RFC 9562 UUIDv8 after setting its version and
  variant fields. The resulting UUID carries 122 digest bits.
- Each hashing helper accepts an OpenSSL algorithm name and defaults to `sha256`. PostgreSQL is
  the authority for which algorithms its linked OpenSSL providers expose, so an unavailable name
  fails in the database instead of being hidden by a stale Python allowlist.
- `sql.days_since` computes fractional age against the database clock.
- `sql.provided` replaces a missing optional expression with typed SQL `NULL`.
- `sql.concat` renders a Python t-string with SQL expressions through `concat`.
- `sql.fragment` renders a nullable t-string fragment that disappears when an interpolation is
  `NULL`.

```python
from sqlalchemy import LargeBinary, column, select

incoming = sql.relation(
    "incoming",
    [column("content", LargeBinary)],
    [(b"first",), (b"second",)],
)

statement = select(sql.uuid8(incoming.c.content, algo="sha256"))
```

These helpers build SQL expressions. They do not open sessions, own engines, or hide transaction
boundaries.
