from importlib.metadata import PackageNotFoundError, version

from .bases import (
    Component,
    FlexModel,
    FrozenFlexModel,
    FrozenModel,
    InternedComponent,
    InternedModelMeta,
    Model,
)
from .cache import DerivedCache
from .decorator import Decorator
from .dispatch import type_dispatch, value_dispatch
from .flyweight import FlyweightMeta
from .lifecycle import IllegalTransition, Lifecycle
from .pipeline import Pipeline, Reversible
from .projection import FieldProjection, Projection
from .registry import Registry, available
from .singleton import Singleton, SingletonMeta
from .strategy import Available, Strategy, StrategyError
from .strflag import StrFlag

try:
    __version__ = version("patos")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Available",
    "available",
    "Component",
    "Decorator",
    "DerivedCache",
    "FieldProjection",
    "FlexModel",
    "FlyweightMeta",
    "FrozenFlexModel",
    "FrozenModel",
    "IllegalTransition",
    "InternedComponent",
    "InternedModelMeta",
    "Lifecycle",
    "Model",
    "Pipeline",
    "Projection",
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
