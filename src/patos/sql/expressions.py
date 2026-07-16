from datetime import datetime
from typing import cast as typing_cast

from sqlalchemy import ColumnElement, Float, extract, func, null, type_coerce


def provided[T](
    column: ColumnElement[T] | ColumnElement[T | None] | None,
) -> ColumnElement[T | None]:
    """Resolve an optional SQL expression against a typed `NULL`."""
    if column is None:
        return typing_cast(ColumnElement[T | None], null())
    return typing_cast(ColumnElement[T | None], column)


def days_since(
    timestamp: ColumnElement[datetime] | ColumnElement[datetime | None],
) -> ColumnElement[float]:
    """Return fractional days from a timestamp to the database clock."""
    one_day = func.make_interval(0, 0, 0, 1)
    age = extract("epoch", func.now() - timestamp) / extract("epoch", one_day)
    return type_coerce(age, Float)
