"""Typed design patterns. The core needs no third-party package.

The pydantic house models live in `patos.models`, installed with `patos[models]`. Their names
(`Model`, `FrozenModel`, `Component`, ...) also resolve from `patos` itself, imported on first use.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import FunctionType
from typing import TYPE_CHECKING

__lazy_modules__ = {
    "patos.cache",
    "patos.decorator",
    "patos.dispatch",
    "patos.flyweight",
    "patos.lifecycle",
    "patos.pipeline",
    "patos.registry",
    "patos.singleton",
    "patos.strategy",
    "patos.strflag",
}

from .cache import DerivedCache, content_key
from .decorator import Decorator
from .dispatch import type_dispatch, value_dispatch
from .flyweight import FlyweightMeta
from .lifecycle import IllegalTransition, Lifecycle
from .pipeline import Pipeline, Reversible
from .registry import Registry, is_available
from .shared import Shared
from .singleton import Singleton, SingletonMeta
from .strategy import Available, Resolution, Strategy, StrategyError
from .strflag import StrFlag

if TYPE_CHECKING:
    # Redundant aliases re-export for the type checkers what `__getattr__` serves at runtime.
    # isort: off
    from .models import (
        Component as Component,
        FieldProjection as FieldProjection,
        FlexModel as FlexModel,
        FrozenFlexModel as FrozenFlexModel,
        FrozenModel as FrozenModel,
        FrozenOpenModel as FrozenOpenModel,
        InternedComponent as InternedComponent,
        InternedModelMeta as InternedModelMeta,
        Model as Model,
        OpenModel as OpenModel,
        Projection as Projection,
        Runtime as Runtime,
        SourceRegistration as SourceRegistration,
        sha256_text_file as sha256_text_file,
        sha256_text_sources as sha256_text_sources,
        statement_bytes as statement_bytes,
    )
    # isort: on

try:
    __version__ = version("patos")
except PackageNotFoundError:
    __version__ = "0.0.0"


_MODELS = frozenset({
    "Component",
    "FieldProjection",
    "FlexModel",
    "FrozenFlexModel",
    "FrozenModel",
    "FrozenOpenModel",
    "InternedComponent",
    "InternedModelMeta",
    "Model",
    "OpenModel",
    "Projection",
    "Runtime",
    "SourceRegistration",
    "sha256_text_file",
    "sha256_text_sources",
    "statement_bytes",
})  # fmt: skip


def __getattr__(name: str) -> type | FunctionType:
    """Resolve a `patos.models` name on first use, so `import patos` never needs pydantic."""
    if name not in _MODELS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".models", __name__), name)


__all__ = [
    "Available",
    "is_available",
    "content_key",
    "Resolution",
    "Shared",
    "Decorator",
    "DerivedCache",
    "FlyweightMeta",
    "IllegalTransition",
    "Lifecycle",
    "Pipeline",
    "Registry",
    "Reversible",
    "Singleton",
    "SingletonMeta",
    "Strategy",
    "StrategyError",
    "StrFlag",
    "type_dispatch",
    "value_dispatch",
]
