from collections.abc import Sequence
from datetime import datetime
from typing import cast
from uuid import UUID

from pydantic import UUID8
from sqlalchemy import ColumnElement, LargeBinary, Text, Uuid, func, type_coerce, values
from sqlalchemy.sql.base import ReadOnlyColumnCollection
from sqlalchemy.sql.elements import ColumnClause, KeyedColumnElement
from sqlalchemy.sql.selectable import CTE

type SQLValue = str | int | float | bool | bytes | datetime | UUID | Sequence[float] | None
type SQLColumn = (
    ColumnClause[str]
    | ColumnClause[int]
    | ColumnClause[float]
    | ColumnClause[bool]
    | ColumnClause[bytes]
    | ColumnClause[datetime]
    | ColumnClause[UUID]
    | ColumnClause[Sequence[float]]
    | ColumnClause[SQLValue]
)
type SQLColumns = ReadOnlyColumnCollection[str, KeyedColumnElement[SQLValue]]


def relation(
    name: str, columns: Sequence[SQLColumn], rows: Sequence[tuple[SQLValue, ...]]
) -> CTE[SQLColumns]:
    """Create a named typed `VALUES` relation for set based PostgreSQL work."""
    return cast(CTE[SQLColumns], values(*columns, name=name).data(rows).cte())


def digest(content: ColumnElement[bytes], algo: str = "sha256") -> ColumnElement[bytes]:
    """Hash bytes in PostgreSQL with an OpenSSL digest selected by name."""
    return type_coerce(func.digest(content, algo), LargeBinary)


def hex(content: ColumnElement[bytes], algo: str = "sha256") -> ColumnElement[str]:
    """Hash bytes in PostgreSQL and return lowercase hexadecimal text."""
    return type_coerce(func.encode(digest(content, algo), "hex"), Text)


def uuid8(content: ColumnElement[bytes], algo: str = "sha256") -> ColumnElement[UUID8]:
    """Hash bytes in PostgreSQL into a deterministic RFC 9562 UUIDv8.

    The first 128 digest bits carry the identity. Six reserved version and variant bits
    are then set, leaving 122 digest bits in the stored UUID.
    """
    raw = func.substr(digest(content, algo), 1, 16)
    versioned = func.set_byte(
        raw,
        6,
        func.get_byte(raw, 6).bitwise_and(0x0F).bitwise_or(0x80),
    )
    canonical = func.set_byte(
        versioned,
        8,
        func.get_byte(raw, 8).bitwise_and(0x3F).bitwise_or(0x80),
    )
    return cast(ColumnElement[UUID8], func.encode(canonical, "hex").cast(Uuid))
