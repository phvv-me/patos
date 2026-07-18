from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Protocol, cast, overload

from pgvector.sqlalchemy import HALFVEC
from pydantic import StringConstraints
from sqlalchemy import ColumnElement, Float, Text
from sqlalchemy.dialects.postgresql import JSONB

from .columns import JSONReader, JSONValue, ScalarReader, ScalarType

NonEmptyString = Annotated[str, StringConstraints(min_length=1)]


class JSONIndexer[T: JSONValue](Protocol):
    """Callable shape of SQLAlchemy's untyped JSON index implementation."""

    def __call__(self, index: str | int) -> ColumnElement[T]: ...


class TypedJSONB(JSONB[JSONValue]):
    """JSONB with typed `>>` text reads and `[kind]` casting readers."""

    cache_ok = True

    class Comparator[T: JSONValue](JSONB.Comparator[T]):
        @overload
        def __getitem__(self, index: type[bool]) -> JSONReader[bool]: ...
        @overload
        def __getitem__(self, index: type[int]) -> JSONReader[int]: ...
        @overload
        def __getitem__(self, index: type[float]) -> JSONReader[float]: ...
        @overload
        def __getitem__(self, index: type[datetime]) -> JSONReader[datetime]: ...
        @overload
        def __getitem__(self, index: str | int) -> ColumnElement[T]: ...
        def __getitem__(self, index: ScalarType | str | int) -> ScalarReader | ColumnElement[T]:
            if isinstance(index, type):
                expression = cast(ColumnElement[JSONValue], self.expr)
                return cast(ScalarReader, JSONReader(expression, index))
            indexer = cast(JSONIndexer[T], super().__getitem__)
            return indexer(index)

        def __rshift__(self, other: str) -> ColumnElement[str]:
            return cast(ColumnElement[str], self.expr.op("->>", return_type=Text)(other))

    comparator_factory = Comparator


class CosineHalfvec(HALFVEC):
    """Half vector whose `@` operator is PostgreSQL cosine distance."""

    cache_ok = True
    render_bind_cast = True

    class Comparator(HALFVEC.Comparator):
        def __matmul__(
            self, other: Sequence[float] | ColumnElement[Sequence[float]]
        ) -> ColumnElement[float]:
            return cast(ColumnElement[float], self.expr.op("<=>", return_type=Float)(other))

    comparator_factory = Comparator
