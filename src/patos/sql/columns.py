from datetime import datetime
from typing import TYPE_CHECKING, Protocol
from typing import cast as typing_cast

from sqlalchemy import Boolean, ColumnElement, DateTime, Float, Integer, Text, cast
from sqlalchemy.types import TypeEngine

type JSONValue = str | int | float | bool | None | dict[str, JSONValue] | list[JSONValue]
type Scalar = int | float | bool | datetime
type ScalarType = type[int] | type[float] | type[bool] | type[datetime]
type CastType = Integer | Float[float] | Boolean | DateTime

_CASTS: dict[ScalarType, CastType] = {
    int: Integer(),
    float: Float(),
    bool: Boolean(),
    datetime: DateTime(timezone=True),
}


class JSONReader[V: Scalar]:
    """Read JSON keys under one Python type and cast their text values in SQL.

    column: JSON column expression to read.
    kind: Python type that selects the SQL cast.
    """

    def __init__(self, column: ColumnElement[JSONValue], kind: type[V]) -> None:
        self.column = column
        self.kind: type[V] = kind

    def __rshift__(self, key: str) -> ColumnElement[V]:
        """Read `key` with `->>` and cast it to this reader's SQL type."""
        reading = self.column.op("->>", return_type=Text)(key)
        kind = typing_cast(ScalarType, self.kind)
        sql_type = typing_cast(TypeEngine[V], _CASTS[kind])
        return typing_cast(ColumnElement[V], cast(reading, sql_type))


type ScalarReader = JSONReader[bool] | JSONReader[int] | JSONReader[float] | JSONReader[datetime]


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import overload

    class ModelInstance(Protocol):
        """Model instance accepted by the descriptor typing facade."""

    class Expr[T](ColumnElement[T]):
        """Typed SQL expression exposed by a `Column[T]` class attribute."""

        @overload
        def __getitem__(self, index: type[bool]) -> JSONReader[bool]: ...
        @overload
        def __getitem__(self, index: type[int]) -> JSONReader[int]: ...
        @overload
        def __getitem__(self, index: type[float]) -> JSONReader[float]: ...
        @overload
        def __getitem__(self, index: type[datetime]) -> JSONReader[datetime]: ...
        @overload
        def __getitem__(self, index: str | int) -> ColumnElement[JSONValue]: ...
        def __getitem__(
            self, index: ScalarType | str | int
        ) -> ScalarReader | ColumnElement[JSONValue]: ...

        def __rshift__(self, other: str) -> ColumnElement[str]: ...

        def __matmul__(
            self,
            other: Sequence[float] | ColumnElement[Sequence[float]] | Expr[Sequence[float]],
        ) -> ColumnElement[float]: ...

    class Column[T]:
        """Annotation facade with values on instances and SQL expressions on classes."""

        @overload
        def __get__(self, instance: None, owner: type[ModelInstance] | None = None) -> Expr[T]: ...
        @overload
        def __get__(
            self, instance: ModelInstance, owner: type[ModelInstance] | None = None
        ) -> T: ...
        def __get__(
            self, instance: ModelInstance | None, owner: type[ModelInstance] | None = None
        ) -> Expr[T] | T: ...

        def __set__(self, instance: ModelInstance, value: T | Expr[T]) -> None: ...

else:
    type Expr[T] = ColumnElement[T]

    class Column:
        """Erase `Column[X]` to `X` so model libraries see the plain annotation."""

        def __class_getitem__[T](cls, item: T) -> T:
            return item
