from typing import Unpack

from pydantic import ConfigDict

from .bases import FrozenModel


class FieldProjection[Projected]:
    """Non-data descriptor projecting one pydantic field at class level.

    Instance attribute lookup finds the validated value in `__dict__` first, so instances
    behave like any pydantic model. Class-level access falls through to this descriptor and
    returns whatever the owning model's `__project__` builds for the field.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(
        self,
        instance: Projection[Projected] | None,
        owner: type[Projection[Projected]],
    ) -> Projected:
        if instance is not None:
            raise AttributeError(self.name)
        return owner.__project__(self.name)


class Projection[Projected](FrozenModel):
    """Immutable model whose fields project into another algebra at class level.

    The generalized hybrid-attribute pattern: `Model.field` on the class returns
    `cls.__project__("field")` while `instance.field` stays the validated value. Subclasses
    implement `__project__` to define the target algebra, a SQL expression, a query fragment,
    a symbolic variable.
    """

    @classmethod
    def __project__(cls, name: str) -> Projected:
        raise NotImplementedError(f"{cls.__name__} does not project field {name!r}")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for name in cls.__pydantic_fields__:
            setattr(cls, name, FieldProjection(name))
