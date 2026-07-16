from collections.abc import Iterable
from enum import EnumType, StrEnum
from typing import cast

from inflection import underscore
from sqlalchemy.dialects.postgresql import ENUM


def enum_values(enum: type[StrEnum]) -> list[str]:
    """Return enum values for SQLAlchemy instead of its default member names."""
    return [str(member.value) for member in enum]


class PGEnumType(EnumType):
    """Expose native PostgreSQL enum metadata on each string enum class."""

    @property
    def name(cls) -> str:
        """Return the canonical PostgreSQL type name from the qualified class name."""
        parts = cls.__qualname__.split(".")
        names = parts[-1:] if "<locals>" in parts else parts
        return "_".join(underscore(part) for part in names)

    @property
    def values(cls) -> tuple[str, ...]:
        """Return the database values in declaration order."""
        members = cast(Iterable[StrEnum], cls)
        return tuple(str(member.value) for member in members)

    @property
    def type(cls) -> ENUM:
        """Return the native SQLAlchemy PostgreSQL enum with its canonical name."""
        return ENUM(cls, name=cls.name, values_callable=enum_values)


class PGEnum(StrEnum, metaclass=PGEnumType):
    """String enum carrying its canonical native PostgreSQL representation."""
