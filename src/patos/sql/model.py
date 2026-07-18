from annotationlib import Format, call_annotate_function, get_annotate_from_class_namespace
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import ClassVar, Protocol, cast, get_origin, overload
from uuid import UUID

import inflection
from sqlalchemy import (
    Boolean,
    ColumnElement,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    Numeric,
    Table,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql.base import RESERVED_WORDS
from sqlalchemy.orm import InstrumentedAttribute, Mapper
from sqlalchemy.orm.properties import ColumnProperty
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.functions import Function
from sqlalchemy.types import TypeEngine
from sqlmodel import SQLModel
from sqlmodel.main import SQLModelMetaclass
from typing_extensions import TypeForm

from .columns import ModelField

_SQL_RESULT_TYPES: dict[type, TypeEngine] = {
    bool: Boolean(),
    bytes: LargeBinary(),
    datetime: DateTime(timezone=True),
    Decimal: Numeric(),
    float: Float(),
    int: Integer(),
    str: Text(),
    UUID: Uuid(),
}

type FunctionArgument = (
    str
    | int
    | float
    | bool
    | bytes
    | date
    | datetime
    | time
    | timedelta
    | UUID
    | ClauseElement
    | None
)


def _table_name(class_name: str) -> str:
    """Derive one safe singular PostgreSQL table name from a model class name."""
    name = inflection.underscore(class_name)
    return f"{name}_" if name in RESERVED_WORDS else name


class ClassMember(Protocol):
    """Accept any class namespace value at the metaclass boundary."""


class SQLFunction[T]:
    """One dynamically named SQL function bound to a column as its first argument."""

    def __init__(self, expression: ClauseElement, name: str) -> None:
        self.expression = expression
        self.name = name

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
    ) -> Function[T] | Function[R]:
        """Call the named SQL function and optionally declare its result type."""
        function = cast(
            Callable[..., Function[T] | Function[R]],
            getattr(func, self.name),
        )
        return (
            function(self.expression, *args)
            if result is None
            else function(self.expression, *args, type_=_SQL_RESULT_TYPES[result])
        )


class SQLFunctions[T]:
    """Typed dynamic function namespace exposed as `Column.f`."""

    def __init__(self, expression: ClauseElement) -> None:
        self.expression = expression

    def __getattr__(self, name: str) -> SQLFunction[T]:
        return SQLFunction(self.expression, name)


class AggregateComparator[T](ColumnProperty.Comparator[T]):
    """Expose common SQL aggregates directly on one mapped column expression."""

    def count(self, distinct: bool = False) -> ColumnElement[int]:
        """Count non-null values, optionally counting distinct values only."""
        expression = self.__clause_element__()
        return func.count(expression.distinct() if distinct else expression)

    @property
    def f(self) -> SQLFunctions[T]:
        """Bind an arbitrary SQL function to this column as its first argument."""
        return SQLFunctions(self.__clause_element__())

    def coalesce(self, *fallbacks: T | ClauseElement) -> ColumnElement[T]:
        """Return the first non-null value from this column and its fallbacks."""
        return cast(ColumnElement[T], func.coalesce(self.__clause_element__(), *fallbacks))

    def greatest(self, *others: T | ClauseElement) -> ColumnElement[T]:
        """Return the greatest value across this column and its peers."""
        return cast(ColumnElement[T], func.greatest(self.__clause_element__(), *others))

    def least(self, *others: T | ClauseElement) -> ColumnElement[T]:
        """Return the least value across this column and its peers."""
        return cast(ColumnElement[T], func.least(self.__clause_element__(), *others))

    def length(self) -> ColumnElement[int]:
        """Return the PostgreSQL length of this column value."""
        return cast(ColumnElement[int], func.length(self.__clause_element__()))

    @overload
    def lower(self) -> ColumnElement[T]: ...

    @overload
    def lower[R](self, *, result: type[R]) -> ColumnElement[R]: ...

    def lower[R](
        self, *, result: type[R] | None = None
    ) -> ColumnElement[T] | ColumnElement[R]:
        """Return the SQL lower value, declaring a different result type when needed."""
        return cast(ColumnElement[T] | ColumnElement[R], func.lower(self.__clause_element__()))

    @overload
    def upper(self) -> ColumnElement[T]: ...

    @overload
    def upper[R](self, *, result: type[R]) -> ColumnElement[R]: ...

    def upper[R](
        self, *, result: type[R] | None = None
    ) -> ColumnElement[T] | ColumnElement[R]:
        """Return the SQL upper value, declaring a different result type when needed."""
        return cast(ColumnElement[T] | ColumnElement[R], func.upper(self.__clause_element__()))

    def sum(self, default: T | None = None) -> ColumnElement[T]:
        """Sum this numeric column and optionally replace an empty result."""
        expression = func.sum(self.__clause_element__())
        return cast(
            ColumnElement[T],
            func.coalesce(expression, default) if default is not None else expression,
        )

    def avg(
        self, default: Decimal | float | None = None
    ) -> ColumnElement[Decimal | float]:
        """Average this numeric column and optionally replace an empty result."""
        expression = func.avg(self.__clause_element__())
        return cast(
            ColumnElement[Decimal | float],
            func.coalesce(expression, default) if default is not None else expression,
        )

    def min(self, default: T | None = None) -> ColumnElement[T]:
        """Return the smallest value and optionally replace an empty result."""
        expression = func.min(self.__clause_element__())
        return cast(
            ColumnElement[T],
            func.coalesce(expression, default) if default is not None else expression,
        )

    def max(self, default: T | None = None) -> ColumnElement[T]:
        """Return the largest value and optionally replace an empty result."""
        expression = func.max(self.__clause_element__())
        return cast(
            ColumnElement[T],
            func.coalesce(expression, default) if default is not None else expression,
        )


class ModelMeta(SQLModelMetaclass):
    """Finish each SQLModel mapping with Patos column expressions."""

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, ClassMember],
        **kwargs: ClassMember,
    ) -> type[SQLModel]:
        if kwargs.get("table") is True and "__tablename__" not in namespace:
            namespace["__tablename__"] = _table_name(name)
        annotations = namespace.get("__annotations__")
        annotate = get_annotate_from_class_namespace(namespace)
        declared = cast(
            dict[str, ClassMember],
            annotations
            if isinstance(annotations, dict)
            else call_annotate_function(annotate, Format.FORWARDREF)
            if annotate is not None
            else {},
        )
        for field_name, value in namespace.items():
            if isinstance(value, ModelField):
                declared[field_name] = value.annotation
                namespace[field_name] = value.info
        for field_name, annotation in declared.items():
            if field_name not in namespace and get_origin(annotation) is not ClassVar:
                namespace[field_name] = ModelField(
                    cast(TypeForm[ClassMember], annotation)
                ).info
        namespace["__annotations__"] = cast(ClassMember, declared)
        return cast(type[SQLModel], super().__new__(cls, name, bases, namespace, **kwargs))

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, ClassMember],
        **kwargs: ClassMember,
    ) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        mapper = cls.__dict__.get("__mapper__")
        if not isinstance(mapper, Mapper):
            return
        for column in mapper.column_attrs:
            attribute = cast(InstrumentedAttribute[ClassMember], getattr(cls, column.key))
            attribute.comparator = AggregateComparator(column, mapper)


class Model(SQLModel, metaclass=ModelMeta):
    """SQLModel base whose mapped columns carry concise typed SQL expressions."""

    __table__: ClassVar[Table]

    @classmethod
    def table_name(cls) -> str:
        """Return the safe singular snake case table name for this model class."""
        return _table_name(cls.__name__)
