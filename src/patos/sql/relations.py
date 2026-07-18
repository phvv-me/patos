import hashlib
from collections.abc import Sequence
from datetime import datetime
from typing import cast, overload
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


@overload
def digest(content: bytes, algo: str = "sha256") -> bytes: ...


@overload
def digest(
    content: ColumnElement[bytes], algo: str = "sha256"
) -> ColumnElement[bytes]: ...


def digest(
    content: bytes | ColumnElement[bytes], algo: str = "sha256"
) -> bytes | ColumnElement[bytes]:
    """Hash raw bytes locally or a byte expression inside PostgreSQL."""
    if isinstance(content, bytes):
        return hashlib.new(algo, content).digest()
    return type_coerce(func.digest(content, algo), LargeBinary)


@overload
def hex(content: bytes, algo: str = "sha256") -> str: ...


@overload
def hex(content: ColumnElement[bytes], algo: str = "sha256") -> ColumnElement[str]: ...


def hex(
    content: bytes | ColumnElement[bytes], algo: str = "sha256"
) -> str | ColumnElement[str]:
    """Hash raw bytes or a PostgreSQL expression into lowercase hexadecimal text."""
    if isinstance(content, bytes):
        return digest(content, algo).hex()
    return type_coerce(func.encode(digest(content, algo), "hex"), Text)


@overload
def uuid8(content: bytes, algo: str = "sha256") -> UUID8: ...


@overload
def uuid8(
    content: ColumnElement[bytes], algo: str = "sha256"
) -> ColumnElement[UUID8]: ...


def uuid8(
    content: bytes | ColumnElement[bytes], algo: str = "sha256"
) -> UUID8 | ColumnElement[UUID8]:
    """Hash raw bytes or a PostgreSQL expression into a deterministic UUIDv8.

    The first 128 digest bits carry the identity. Six reserved version and variant bits
    are then set, leaving 122 digest bits in the stored UUID.
    """
    if isinstance(content, bytes):
        return UUID(bytes=digest(content, algo)[:16], version=8)
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
