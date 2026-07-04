import abc
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from .flyweight import FlyweightMeta
from .registry import Registry

if TYPE_CHECKING:
    # Pydantic does not export its model metaclass publicly, so the type checker sees it
    # through its internal module while runtime keeps resolving it dynamically below. This
    # gives mypy a concrete base for `InternedModelMeta`, which both clears the "dynamic base
    # class" error on the expression it replaces and lets mypy see `InternedModelMeta` as an
    # actual `ModelMetaclass` subclass, resolving the metaclass conflict on `InternedComponent`.
    from pydantic._internal._model_construction import ModelMetaclass
else:
    ModelMetaclass = type(BaseModel)

IGNORED_TYPES: tuple[type, ...] = (cached_property,)


class Model(BaseModel):
    """Mutable model with standard types only.

    Use for simple value objects, result structs, game entities, etc.
    """

    model_config = ConfigDict(ignored_types=IGNORED_TYPES)


class FrozenModel(BaseModel):
    """Immutable model with standard types only.

    Use for configuration objects and AST nodes that should never be mutated after
    construction.
    """

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        ignored_types=IGNORED_TYPES,
    )


class FlexModel(BaseModel):
    """Mutable model that accepts arbitrary types (tensors, tokenizers, etc.).

    Use when fields include `torch.Tensor`, `numpy.ndarray`, `PreTrainedModel`, or other
    types pydantic cannot validate natively.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=IGNORED_TYPES,
    )


class FrozenFlexModel(BaseModel):
    """Immutable model that accepts arbitrary types.

    Combines `frozen=True` with `arbitrary_types_allowed=True`. Use for frozen configs or
    tables that hold tensor data.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        ignored_types=IGNORED_TYPES,
    )


class InternedModelMeta(FlyweightMeta, ModelMetaclass):
    """Pydantic model metaclass that also interns instances by construction arguments.

    Combines `FlyweightMeta` with pydantic's own metaclass so a frozen model becomes a
    flyweight: identical construction arguments return the same object, and every
    `cached_property` is therefore computed once per distinct configuration.
    """


class Component(Registry, FrozenFlexModel, abc.ABC):
    """The shared spine for every named, frozen-config object (codecs, lattices, gauges).

    A subclass gets self-registration with an auto kebab-case `name` (the `Registry`
    `find`/`implementations`/`dispatch` surface) and an immutable arbitrary-type pydantic
    config, so it only declares its hyperparameter fields, its derived tensors as
    `functools.cached_property`, and its one contract method (`encode`/`decode`, `nearest`,
    `apply` ...).
    """


class InternedComponent(Component, metaclass=InternedModelMeta):
    """A `Component` whose identical-config instances are interned (flyweight).

    `E8()` is always the same object, so cached-property tables build once per distinct
    configuration. Declare interning by subclassing this instead of `Component`.
    """
