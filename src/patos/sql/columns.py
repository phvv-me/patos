import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from decimal import Decimal
from operator import or_
from types import NoneType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    Protocol,
    Union,
    get_args,
    get_origin,
)
from typing import cast as typing_cast

from pydantic import UUID7
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined, PydanticUndefinedType
from sqlalchemy import Boolean, ColumnElement, DateTime, Float, Integer, Text, cast, func
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement
from sqlalchemy.types import TypeEngine
from sqlmodel import Field as SQLModelField
from typing_extensions import Sentinel, TypeForm

from .enum import PGEnum

_INFER_SERVER_DEFAULT = Sentinel("_INFER_SERVER_DEFAULT")

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


class TypeExpression(Protocol):
    """Runtime type expression at the checker boundary for equality tests."""


class SQLTypeExpression(Protocol):
    """SQLAlchemy type accepted at the SQLModel stub boundary."""


class FunctionArgument(Protocol):
    """Value or SQL expression accepted by a dynamically named function."""


class SQLModelFieldFactory(Protocol):
    """Typed view of the SQLModel field options Patos centralizes."""

    def __call__[T](
        self,
        default: T | PydanticUndefinedType = PydanticUndefined,
        *,
        default_factory: Callable[[], T] | None = None,
        primary_key: bool | PydanticUndefinedType = PydanticUndefined,
        foreign_key: str | PydanticUndefinedType = PydanticUndefined,
        ondelete: Literal["CASCADE", "SET NULL", "RESTRICT"]
        | PydanticUndefinedType = PydanticUndefined,
        unique: bool | PydanticUndefinedType = PydanticUndefined,
        nullable: bool | PydanticUndefinedType = PydanticUndefined,
        index: bool | PydanticUndefinedType = PydanticUndefined,
        ge: int | float | None = None,
        gt: int | float | None = None,
        le: int | float | None = None,
        lt: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        sa_type: SQLTypeExpression | PydanticUndefinedType = PydanticUndefined,
        sa_column_kwargs: Mapping[str, str | ClauseElement]
        | PydanticUndefinedType = PydanticUndefined,
    ) -> FieldInfo: ...


def concrete_type(kind: TypeExpression) -> type | None:
    """Find the concrete Python type inside annotations and optional unions."""
    origin = get_origin(kind)
    if origin is Annotated:
        return concrete_type(typing_cast(TypeExpression, get_args(kind)[0]))
    if origin in (Union, UnionType):
        candidates = [candidate for candidate in get_args(kind) if candidate is not NoneType]
        return concrete_type(typing_cast(TypeExpression, candidates[0])) if candidates else None
    if isinstance(origin, type):
        return origin
    return kind if isinstance(kind, type) else None


def allows_none(kind: TypeExpression) -> bool:
    """Return whether an annotation explicitly accepts `None`."""
    origin = get_origin(kind)
    if origin is Annotated:
        return allows_none(typing_cast(TypeExpression, get_args(kind)[0]))
    return origin in (Union, UnionType) and NoneType in get_args(kind)


def inferred_sql_type(
    kind: TypeExpression,
    max_length: int | None = None,
) -> SQLTypeExpression | None:
    """Map annotations with PostgreSQL semantics that SQLModel cannot infer itself."""
    concrete = concrete_type(kind)
    if concrete is not None and issubclass(concrete, PGEnum):
        return concrete.type
    if concrete is datetime:
        return DateTime(timezone=True)
    if concrete is str and max_length is None:
        return Text
    return None


def inferred_server_default[RowT](
    kind: TypeExpression,
    default: RowT | PydanticUndefinedType,
    default_factory: Callable[[], RowT] | None,
) -> str | ClauseElement | None:
    """Derive a database default when the Python declaration is unambiguous."""
    if not isinstance(default, PydanticUndefinedType) and default is not None:
        if isinstance(default, bool):
            return str(default).lower()
        if isinstance(default, int | float | str):
            return str(default)
    concrete = concrete_type(kind)
    if concrete is datetime and default_factory is not None:
        return func.now()
    if concrete is dict and default_factory is dict:
        return "{}"
    return None


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

    from sqlalchemy.sql.functions import Function

    class ModelInstance(Protocol):
        """Model instance accepted by the descriptor typing facade."""

    class Expr[T](ColumnElement[T]):
        """Typed SQL expression exposed by a `Column[T]` class attribute."""

        def count(self, distinct: bool = False) -> ColumnElement[int]: ...
        def sum(self, default: T | None = None) -> ColumnElement[T]: ...
        def avg(
            self, default: Decimal | float | None = None
        ) -> ColumnElement[Decimal | float]: ...
        def min(self, default: T | None = None) -> ColumnElement[T]: ...
        def max(self, default: T | None = None) -> ColumnElement[T]: ...

        def coalesce(self, *fallbacks: T | ClauseElement) -> ColumnElement[T]: ...
        def greatest(self, *others: T | ClauseElement) -> ColumnElement[T]: ...
        def least(self, *others: T | ClauseElement) -> ColumnElement[T]: ...
        def length(self) -> ColumnElement[int]: ...
        @overload
        def lower(self) -> ColumnElement[T]: ...
        @overload
        def lower[R](self, *, result: type[R]) -> ColumnElement[R]: ...
        def lower[R](
            self, *, result: type[R] | None = None
        ) -> ColumnElement[T] | ColumnElement[R]: ...

        @overload
        def upper(self) -> ColumnElement[T]: ...
        @overload
        def upper[R](self, *, result: type[R]) -> ColumnElement[R]: ...
        def upper[R](
            self, *, result: type[R] | None = None
        ) -> ColumnElement[T] | ColumnElement[R]: ...

        @property
        def f(self) -> SQLFunctions[T]: ...

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

    class SQLFunction[T]:
        """One dynamically named SQL function bound to a column as its first argument."""

        @overload
        def __call__(
            self,
            *args: FunctionArgument,
            result: None = None,
        ) -> Function[T]: ...
        @overload
        def __call__[R](
            self,
            *args: FunctionArgument,
            result: type[R],
        ) -> Function[R]: ...
        def __call__[R](
            self,
            *args: FunctionArgument,
            result: type[R] | None = None,
        ) -> Function[T] | Function[R]: ...

    class SQLFunctions[T]:
        """Typed dynamic function namespace exposed as `Column.f`."""

        def __getattr__(self, name: str) -> SQLFunction[T]: ...

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


class ModelField[T]:
    """Build one inferred SQLModel field for Patos model descriptors."""

    def __init__(
        self,
        kind: TypeForm[T],
        *,
        default: T | None | PydanticUndefinedType = PydanticUndefined,
        default_factory: Callable[[], T] | None = None,
        primary_key: bool | None = None,
        foreign_key: str | None = None,
        ondelete: Literal["CASCADE", "SET NULL", "RESTRICT"] | None = None,
        unique: bool | None = None,
        nullable: bool | None = None,
        index: bool | None = None,
        ge: int | float | None = None,
        gt: int | float | None = None,
        le: int | float | None = None,
        lt: int | float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        sa_type: SQLTypeExpression | None = None,
        server_default: str | ClauseElement | None | _INFER_SERVER_DEFAULT = _INFER_SERVER_DEFAULT,
        onupdate: str | ClauseElement | None = None,
    ) -> None:
        self.annotation = typing_cast(TypeExpression, kind)
        resolved_python_default = (
            None
            if isinstance(default, PydanticUndefinedType)
            and default_factory is None
            and allows_none(self.annotation)
            else default
        )
        column_options: dict[str, str | ClauseElement] = {}
        resolved_default = (
            inferred_server_default(self.annotation, resolved_python_default, default_factory)
            if server_default is _INFER_SERVER_DEFAULT
            else server_default
        )
        if resolved_default is not None:
            column_options["server_default"] = resolved_default
        if onupdate is not None:
            column_options["onupdate"] = onupdate
        resolved = sa_type or inferred_sql_type(self.annotation, max_length)
        factory = typing_cast(SQLModelFieldFactory, SQLModelField)
        generated = factory(
            resolved_python_default,
            default_factory=default_factory,
            primary_key=PydanticUndefined if primary_key is None else primary_key,
            foreign_key=PydanticUndefined if foreign_key is None else foreign_key,
            ondelete=PydanticUndefined if ondelete is None else ondelete,
            unique=PydanticUndefined if unique is None else unique,
            nullable=PydanticUndefined if nullable is None else nullable,
            index=PydanticUndefined if index is None else index,
            ge=ge,
            gt=gt,
            le=le,
            lt=lt,
            min_length=min_length,
            max_length=max_length,
            sa_type=PydanticUndefined if resolved is None else resolved,
            sa_column_kwargs=column_options or PydanticUndefined,
        )
        self.info = generated


class Field[T](ModelField[T]):
    """Expose one SQL type on the class and its Python value on instances."""

    if TYPE_CHECKING:

        @overload
        def __get__(
            self, instance: None, owner: type[ModelInstance] | None = None
        ) -> Expr[T]: ...

        @overload
        def __get__(
            self, instance: ModelInstance, owner: type[ModelInstance] | None = None
        ) -> T: ...
        def __get__(
            self, instance: ModelInstance | None, owner: type[ModelInstance] | None = None
        ) -> Expr[T] | T: ...

        def __set__(self, instance: ModelInstance, value: T | Expr[T]) -> None: ...


class GeneratedField[T](ModelField[T]):
    """Expose a generated SQL value that is absent before its row is inserted."""

    if TYPE_CHECKING:

        @overload
        def __get__(
            self, instance: None, owner: type[ModelInstance] | None = None
        ) -> Expr[T]: ...

        @overload
        def __get__(
            self, instance: ModelInstance, owner: type[ModelInstance] | None = None
        ) -> T | None: ...
        def __get__(
            self, instance: ModelInstance | None, owner: type[ModelInstance] | None = None
        ) -> Expr[T] | T | None: ...

        def __set__(self, instance: ModelInstance, value: T | Expr[T] | None) -> None: ...


def declared_type[T](target: Expr[T]) -> TypeForm[T]:
    """Recover a mapped attribute's original Python annotation from its declaring model."""
    mapped = typing_cast(InstrumentedAttribute[T], target)
    for owner in mapped.class_.__mro__:
        annotations = typing_cast(
            Mapping[str, TypeForm[T]],
            owner.__dict__.get("__annotations__", {}),
        )
        if mapped.key in annotations:
            return annotations[mapped.key]
    raise TypeError(f"{mapped.class_.__name__}.{mapped.key} has no declared type")


if TYPE_CHECKING:

    @overload
    def FK[T](
        target: Expr[T],
        *,
        nullable: Literal[False] = False,
        ondelete: Literal["CASCADE", "SET NULL", "RESTRICT"] | None = None,
        unique: bool | None = None,
        index: bool | None = None,
    ) -> Field[T]: ...

    @overload
    def FK[T](
        target: Expr[T],
        *,
        nullable: Literal[True],
        ondelete: Literal["CASCADE", "SET NULL", "RESTRICT"] | None = None,
        unique: bool | None = None,
        index: bool | None = None,
    ) -> Field[T | None]: ...


def FK[T](
    target: Expr[T],
    *,
    nullable: bool = False,
    ondelete: Literal["CASCADE", "SET NULL", "RESTRICT"] | None = None,
    unique: bool | None = None,
    index: bool | None = None,
) -> Field[T] | Field[T | None]:
    """Declare a scalar foreign key from one mapped primary or unique column."""
    mapped = typing_cast(InstrumentedAttribute[T], target)
    remote = mapped.property.columns[0]
    if not remote.primary_key and not remote.unique:
        raise ValueError(f"{remote} must be primary or unique to be a foreign-key target")
    kind = declared_type(target)
    foreign_key = f"{remote.table.fullname}.{remote.name}"
    if nullable:
        optional = typing_cast(TypeForm[T | None], or_(kind, None))
        return Field(
            optional,
            foreign_key=foreign_key,
            ondelete=ondelete,
            index=index,
            unique=unique,
        )
    return Field(
        kind,
        foreign_key=foreign_key,
        ondelete=ondelete,
        index=index,
        unique=unique,
    )


if TYPE_CHECKING:

    @overload
    def PK(kind: type[int]) -> GeneratedField[int]: ...

    @overload
    def PK[T](kind: TypeForm[T]) -> Field[T]: ...


def PK[T](kind: TypeForm[T]) -> Field[T] | GeneratedField[int]:
    """Declare one generated or required primary key as a typed descriptor."""
    expression = typing_cast(TypeExpression, kind)
    if expression == typing_cast(TypeExpression, UUID7):
        factory = typing_cast(Callable[[], T], uuid.uuid7)
        return Field(kind, default_factory=factory, primary_key=True)
    if expression is int:
        return GeneratedField(
            typing_cast(TypeForm[int], kind), default=None, primary_key=True
        )
    if expression is str:
        return Field(kind, primary_key=True, sa_type=Text)
    return Field(kind, primary_key=True)


def Nullable[T](kind: TypeForm[T]) -> Field[T | None]:
    """Declare an optional column whose inferred default is `None`."""
    annotation = typing_cast(TypeForm[T | None], or_(kind, None))
    return Field(annotation)
