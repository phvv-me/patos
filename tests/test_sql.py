from datetime import datetime
from enum import auto
from typing import cast

import pytest
import sqlalchemy as sa
from hypothesis import given
from hypothesis import strategies as st
from pydantic import UUID7, UUID8, ValidationError
from sqlalchemy import ColumnElement
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import DefaultClause
from sqlalchemy.sql import ClauseElement
from sqlmodel import Field

from patos import sql


class Doc(sql.Model, table=True):
    id = sql.PK(int)
    title: sql.Column[str]
    summary = sql.Nullable(str)
    category = sql.Field(str)
    attributes: sql.Column[dict] = Field(default_factory=dict, sa_type=sql.TypedJSONB)
    embedding: sql.Column[list[float] | None] = Field(
        default=None, sa_type=cast(type[list[float]], sql.CosineHalfvec(3))
    )
    created_at: sql.Column[datetime]


def compiled(expression: ClauseElement) -> str:
    """Compile one SQL expression against PostgreSQL with inline literals."""
    return str(
        expression.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )


def test_column_facade_erases_at_runtime_and_preserves_plain_values() -> None:
    assert sql.Column[dict] is dict
    assert sql.Column[list[float] | None] == list[float] | None
    assert sql.Expr[int]
    doc = Doc(
        title="a title",
        category="reference",
        attributes={"summary": "text"},
        created_at=datetime(2026, 1, 1),
    )
    assert doc.title == "a title"
    assert doc.summary is None
    assert doc.category == "reference"
    assert doc.attributes == {"summary": "text"}
    assert compiled(Doc.title) == "doc.title"
    assert compiled(Doc.summary) == "doc.summary"
    assert compiled(Doc.category) == "doc.category"
    assert isinstance(Doc.__table__.c.title.type, sa.Text)
    assert isinstance(Doc.__table__.c.created_at.type, sa.DateTime)
    assert Doc.__table__.c.created_at.type.timezone


def test_model_derives_safe_singular_snake_case_table_names() -> None:
    class AuditEvent(sql.Model, table=True):
        id = sql.PK(int)

    class User(sql.Model, table=True):
        id = sql.PK(int)

    assert Doc.__tablename__ == "doc"
    assert AuditEvent.__tablename__ == "audit_event"
    assert User.__tablename__ == "user_"
    assert AuditEvent.table_name() == "audit_event"


def test_primary_key_facade_encodes_generation_policy() -> None:
    class Generated(sql.Model):
        id = sql.PK(UUID7)

    class Inherited(Generated, table=True):
        name: sql.Column[str]

    class Event(sql.Model, table=True):
        id = sql.PK(UUID8)

    class Capture(sql.Model, table=True):
        id = sql.PK(UUID7)

    class Label(sql.Model, table=True):
        name = sql.PK(str)

    assert Doc.__table__.c.id.primary_key
    assert Doc(title="", created_at=datetime(2026, 1, 1)).id is None
    assert Inherited(name="inherited").id.version == 7
    assert Inherited.__table__.c.id.primary_key
    assert Event.__table__.c.id.primary_key
    assert Capture().id.version == 7
    assert isinstance(Label.__table__.c.name.type, sa.Text)


def test_foreign_key_facade_derives_target_type_and_constraint() -> None:
    class Category(sql.Model, table=True):
        name = sql.PK(str)

    class Entry(sql.Model, table=True):
        id = sql.PK(int)
        category = sql.FK(Category.name, ondelete="CASCADE", index=True)
        fallback = sql.FK(Category.name, nullable=True)

    entry = Entry(category="paper")
    category = Entry.__table__.c.category
    fallback = Entry.__table__.c.fallback

    assert entry.category == "paper"
    assert entry.fallback is None
    assert category.foreign_keys.pop().target_fullname == "category.name"
    assert category.index
    assert fallback.nullable


@pytest.mark.parametrize(
    ("value_type", "key", "expected", "sql_type", "timezone"),
    [
        (int, "count", "CAST(doc.attributes ->> 'count' AS INTEGER)", sa.Integer, None),
        (
            datetime,
            "expires_at",
            "CAST(doc.attributes ->> 'expires_at' AS TIMESTAMP WITH TIME ZONE)",
            sa.DateTime,
            True,
        ),
    ],
    ids=["integer", "timestamp"],
)
def test_reader_casts_text_lookups_to_the_requested_sql_type(
    value_type: type[int] | type[datetime],
    key: str,
    expected: str,
    sql_type: type[sa.types.TypeEngine],
    timezone: bool | None,
) -> None:
    expression = Doc.attributes[value_type] >> key
    assert compiled(expression) == expected
    assert isinstance(expression.type, sql_type)
    if timezone is not None:
        assert isinstance(expression.type, sa.DateTime)
        assert expression.type.timezone is timezone


def test_expression_and_template_helpers_compile_complete_contracts() -> None:
    assert sql.provided(Doc.title) is Doc.title
    assert compiled(sql.provided(None)) == "NULL"
    assert compiled(sql.days_since(Doc.created_at)) == (
        "EXTRACT(epoch FROM now() - doc.created_at)"
        " / CAST(EXTRACT(epoch FROM make_interval(0, 0, 0, 1)) AS NUMERIC)"
    )
    title, summary = Doc.title, Doc.summary
    assert compiled(sql.concat(t"title: {title}, summary: {summary}")) == (
        "concat('title: ', doc.title, ', summary: ', doc.summary)"
    )
    assert compiled(sql.fragment(t"summary: {summary}")) == (
        "coalesce('summary: ' || doc.summary, '')"
    )


def test_model_columns_expose_typed_aggregate_expressions() -> None:
    assert compiled(Doc.id.count()) == "count(doc.id)"
    assert compiled(Doc.id.count(distinct=True)) == "count(DISTINCT doc.id)"
    assert compiled(Doc.id.sum()) == "sum(doc.id)"
    assert compiled(Doc.id.avg()) == "avg(doc.id)"
    assert compiled(Doc.id.min()) == "min(doc.id)"
    assert compiled(Doc.id.max()) == "max(doc.id)"
    assert compiled(Doc.id.sum(default=0)) == "coalesce(sum(doc.id), 0)"
    assert compiled(Doc.title.lower()) == "lower(doc.title)"
    assert compiled(Doc.title.length()) == "length(doc.title)"
    assert compiled(Doc.title.coalesce("untitled")) == "coalesce(doc.title, 'untitled')"
    assert compiled(Doc.title.coalesce(sa.column("fallback"))) == (
        "coalesce(doc.title, fallback)"
    )
    assert compiled(Doc.id.greatest(0)) == "greatest(doc.id, 0)"
    assert compiled(Doc.id.least(10)) == "least(doc.id, 10)"
    assert compiled(Doc.title.f.substr(1, 3)) == "substr(doc.title, 1, 3)"
    position = Doc.title.f.strpos("title", result=int)
    assert compiled(position) == "strpos(doc.title, 'title')"
    assert isinstance(position.type, sa.Integer)
    values = Doc.embedding.f.unnest().table_valued("value").render_derived()
    assert compiled(sa.select(values.c.value)) == (
        "SELECT anon_1.value \nFROM unnest(doc.embedding) AS anon_1(value)"
    )


@given(key=st.sampled_from(("summary", "source", "an odd key")))
def test_postgresql_types_expose_concise_operators(key: str) -> None:
    extracted = Doc.attributes >> key
    assert compiled(extracted) == f"doc.attributes ->> '{key}'"
    assert isinstance(extracted.type, sa.Text)
    assert compiled(Doc.attributes[key]) == f"doc.attributes['{key}']"
    distance = Doc.embedding @ sa.column("query_vector")
    assert compiled(distance) == "doc.embedding <=> query_vector"
    assert isinstance(distance.type, sa.Float)


@pytest.mark.parametrize("algo", ["sha256", "sha3-256", "blake2s256"])
def test_relations_and_database_hashing_compile_as_set_based_sql(
    algo: str,
) -> None:
    incoming = sql.relation(
        "incoming",
        [sa.column("content", sa.LargeBinary)],
        [(b"hello",)],
    )
    assert "VALUES ('hello')" in compiled(sa.select(incoming.c.content))
    content = cast(ColumnElement[bytes], incoming.c.content)
    assert compiled(sql.digest(content, algo=algo)) == f"digest(incoming_1.content, '{algo}')"
    assert compiled(sql.hex(content, algo=algo)) == (
        f"encode(digest(incoming_1.content, '{algo}'), 'hex')"
    )
    uuid = sql.uuid8(content, algo=algo)
    assert isinstance(uuid.type, sa.Uuid)
    assert compiled(uuid) == (
        "CAST(encode(set_byte(set_byte(substr(digest(incoming_1.content, "
        f"'{algo}'), 1, 16), 6, (get_byte(substr(digest(incoming_1.content, '{algo}'), "
        "1, 16), 6) & 15) | 128), 8, (get_byte(substr(digest(incoming_1.content, "
        f"'{algo}'), 1, 16), 8) & 63) | 128), 'hex') AS UUID)"
    )


def test_uuid8_helper_preserves_pydantic_typing() -> None:
    local: UUID8 = sql.uuid8(b"hello")
    expression: ColumnElement[UUID8] = sql.uuid8(
        cast(ColumnElement[bytes], sa.column("content", sa.LargeBinary))
    )

    assert sql.digest(b"hello") == bytes.fromhex(
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert sql.hex(b"hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert str(local) == "2cf24dba-5fb0-830e-a6e8-3b2ac5b9e29e"
    assert isinstance(expression.type, sa.Uuid)


class Watermark:
    class Kind(sql.PGEnum):
        ready = auto()
        active = "open"


def test_pg_enum_uses_qualified_names_and_persists_values() -> None:
    pg = Watermark.Kind.type

    assert Watermark.Kind.name == "watermark_kind"
    assert Watermark.Kind.values == ("ready", "open")
    assert pg.name == "watermark_kind"
    assert pg.enums == ["ready", "open"]
    assert Watermark.Kind.ready.name == "ready"


def test_field_infers_constraints_timestamps_and_native_enums() -> None:
    class ObjectState(sql.PGEnum):
        ready = auto()
        failed = auto()

    class StoredObject(sql.Model, table=True):
        id = sql.PK(int)
        size: sql.Column[sql.NonNegativeInt]
        key = sql.Field(sql.NonEmptyString, max_length=12, unique=True)
        state = sql.Field(ObjectState, default=ObjectState.ready, index=True)
        checked_at = sql.Nullable(datetime)
        inferred_optional = sql.Field(str | None)
        created_at = sql.Field(datetime, default_factory=datetime.now)
        details = sql.Field(dict, default_factory=dict, sa_type=sql.TypedJSONB)
        explicit_only = sql.Field(int, default=1, server_default=None)

    valid = StoredObject.model_validate({"size": 0, "key": "object-key"})
    assert valid.state is ObjectState.ready
    assert StoredObject.__table__.c.key.type.compile(dialect=postgresql.dialect()) == "VARCHAR(12)"
    assert StoredObject.__table__.c.key.unique
    assert isinstance(StoredObject.__table__.c.state.type, postgresql.ENUM)
    assert StoredObject.__table__.c.state.type.name == "object_state"
    assert StoredObject.__table__.c.state.index
    assert isinstance(StoredObject.__table__.c.state.server_default, DefaultClause)
    assert str(StoredObject.__table__.c.state.server_default.arg) == "ready"
    assert isinstance(StoredObject.__table__.c.checked_at.type, sa.DateTime)
    assert StoredObject.__table__.c.checked_at.type.timezone
    assert valid.checked_at is None
    assert valid.inferred_optional is None
    assert StoredObject.__table__.c.inferred_optional.nullable
    assert StoredObject.__table__.c.created_at.server_default is not None
    assert isinstance(StoredObject.__table__.c.details.server_default, DefaultClause)
    assert str(StoredObject.__table__.c.details.server_default.arg) == "{}"
    assert StoredObject.__table__.c.explicit_only.server_default is None
    with pytest.raises(ValidationError):
        StoredObject.model_validate({"size": -1, "key": ""})
