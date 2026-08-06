import abc
import hashlib
import json
from functools import cached_property
from typing import TYPE_CHECKING, Annotated

from pydantic import ConfigDict, GetCoreSchemaHandler, GetPydanticSchema
from pydantic.main import BaseModel
from pydantic_core import core_schema

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

_ignored_types: tuple[type, ...] = (cached_property,)


def _runtime_schema(
    source_type: type,
    handler: GetCoreSchemaHandler,
) -> core_schema.CoreSchema:
    """Treat one already-validated live object as an opaque runtime value."""
    return core_schema.any_schema()


type Runtime[T] = Annotated[T, GetPydanticSchema(_runtime_schema)]


class Model(BaseModel):
    """Mutable model with standard types only.

    Use for simple value objects, result structs, game entities, etc.
    """

    model_config = ConfigDict(extra="forbid", ignored_types=_ignored_types)


class FrozenModel(BaseModel):
    """Immutable model with standard types only.

    Use for configuration objects and AST nodes that should never be mutated after
    construction.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        ignored_types=_ignored_types,
    )

    @cached_property
    def stable_id(self) -> int:
        """Return a deterministic 64-bit identity for this model and its validated fields."""
        model = self.__class__
        payload = {
            "model": f"{model.__module__}.{model.__qualname__}",
            "fields": self.model_dump(mode="json", round_trip=True),
        }
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        digest = hashlib.blake2b(canonical, digest_size=8, person=b"patos-id").digest()
        return int.from_bytes(digest)


class OpenModel(Model):
    """Mutable model over a payload somebody else authors, keeping only the declared fields.

    Use this exactly when the data comes from outside and its shape is not yours to fix, such
    as an OIDC discovery document, an OpenAI-compatible response, or any REST provider. Those
    formats are specified as open, so a provider may advertise more than a given reader
    consumes and adding a field is a compatible change on its side. `Model` forbids extras
    because a payload we author carries exactly what it declares, and a stray key there is a
    typo worth failing on. Applying that same rule to somebody else's payload converts every
    upstream addition into an outage, so the open bases exist to make the distinction a choice
    of base class rather than a config incantation each model has to remember.

    Unknown fields are dropped rather than kept, so the parsed object still carries exactly the
    declared fields and nothing downstream can come to depend on a key the provider never
    promised.
    """

    model_config = ConfigDict(extra="ignore")


class FrozenOpenModel(FrozenModel):
    """Immutable model over a payload somebody else authors, keeping only the declared fields.

    The frozen half of `OpenModel`, and the one most integrations want, because a parsed
    provider response is a record rather than a workspace. Dropping unknown fields also keeps
    `stable_id` a function of what this model declares, so the identity of an unchanged
    response survives the provider adding metadata beside it.
    """

    model_config = ConfigDict(extra="ignore")


class FlexModel(BaseModel):
    """Mutable model that accepts arbitrary types (tensors, tokenizers, etc.).

    Use when fields include `torch.Tensor`, `numpy.ndarray`, `PreTrainedModel`, or other
    types pydantic cannot validate natively. Flexibility here is about the types a field may
    hold and never about accepting undeclared fields, which is what the open bases do.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=_ignored_types,
    )


class FrozenFlexModel(FrozenModel):
    """Immutable model that accepts arbitrary types.

    Extends `FrozenModel` with arbitrary type support. Use when arbitrary values are the model's
    intended data contract. Prefer field-local `Runtime[T]` on `FrozenModel` when only selected
    fields hold already-validated callables, locks, syntax trees, tensors, or clients.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
