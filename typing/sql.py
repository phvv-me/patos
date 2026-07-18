from datetime import datetime
from typing import assert_type

from patos import sql
from patos.sql import Column as C


class Record(sql.Model, table=True):
    id = sql.PK(int)
    name = sql.Field(str)
    note = sql.Nullable(str)
    created_at: C[datetime]


class Label(sql.Model, table=True):
    name = sql.PK(str)


class TaggedRecord(sql.Model, table=True):
    id = sql.PK(int)
    label = sql.FK(Label.name)
    fallback = sql.FK(Label.name, nullable=True)


record = Record(name="typed", created_at=datetime(2026, 7, 16))

assert_type(Record.id, sql.Expr[int])
assert_type(record.id, int | None)
assert_type(Record.name, sql.Expr[str])
assert_type(record.name, str)
assert_type(Record.note, sql.Expr[str | None])
assert_type(record.note, str | None)
assert_type(Record.created_at, sql.Expr[datetime])
assert_type(record.created_at, datetime)
assert_type(TaggedRecord.label, sql.Expr[str])
assert_type(TaggedRecord(label="typed").label, str)
assert_type(TaggedRecord.fallback, sql.Expr[str | None])
assert_type(TaggedRecord(label="typed").fallback, str | None)
