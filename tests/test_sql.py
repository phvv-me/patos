from datetime import datetime
from enum import auto
from typing import cast

import pytest
import sqlalchemy as sa
from hypothesis import given
from hypothesis import strategies as st
from pydantic import UUID8
from sqlalchemy import ColumnElement
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import ClauseElement
from sqlmodel import Field, SQLModel

from patos import sql


class Doc(SQLModel, table=True):
    id: sql.Column[int] = Field(default=None, primary_key=True)
    title: sql.Column[str]
    summary: sql.Column[str | None] = Field(default=None)
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
    doc = Doc(title="a title", attributes={"summary": "text"}, created_at=datetime(2026, 1, 1))
    assert doc.title == "a title"
    assert doc.attributes == {"summary": "text"}
    assert compiled(Doc.title) == "doc.title"


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
    expression: ColumnElement[UUID8] = sql.uuid8(
        cast(ColumnElement[bytes], sa.column("content", sa.LargeBinary))
    )
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
