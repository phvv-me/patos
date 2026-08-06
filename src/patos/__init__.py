from importlib.metadata import PackageNotFoundError, version

__lazy_modules__ = {
    "patos.bases",
    "patos.cache",
    "patos.decorator",
    "patos.dispatch",
    "patos.flyweight",
    "patos.lifecycle",
    "patos.pipeline",
    "patos.projection",
    "patos.registry",
    "patos.singleton",
    "patos.strategy",
    "patos.strflag",
}

from .bases import (
    Component,
    FlexModel,
    FrozenFlexModel,
    FrozenModel,
    FrozenOpenModel,
    InternedComponent,
    InternedModelMeta,
    Model,
    OpenModel,
    Runtime,
)
from .cache import DerivedCache
from .decorator import Decorator
from .dispatch import type_dispatch, value_dispatch
from .flyweight import FlyweightMeta
from .lifecycle import IllegalTransition, Lifecycle
from .pipeline import Pipeline, Reversible
from .projection import FieldProjection, Projection
from .registry import Registry, is_available
from .singleton import Singleton, SingletonMeta
from .strategy import Available, Strategy, StrategyError
from .strflag import StrFlag

try:
    __version__ = version("patos")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Available",
    "is_available",
    "Component",
    "Decorator",
    "DerivedCache",
    "FieldProjection",
    "FlexModel",
    "FlyweightMeta",
    "FrozenFlexModel",
    "FrozenModel",
    "FrozenOpenModel",
    "IllegalTransition",
    "InternedComponent",
    "InternedModelMeta",
    "Lifecycle",
    "Model",
    "OpenModel",
    "Pipeline",
    "Projection",
    "Registry",
    "Reversible",
    "Runtime",
    "Singleton",
    "SingletonMeta",
    "Strategy",
    "StrategyError",
    "StrFlag",
    "type_dispatch",
    "value_dispatch",
]
