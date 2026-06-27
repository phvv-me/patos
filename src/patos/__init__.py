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
    "Component",
    "Decorator",
    "DerivedCache",
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
    "Registry",
    "Reversible",
    "Singleton",
    "SingletonMeta",
    "StrFlag",
    "Strategy",
    "StrategyError",
    "available",
    "type_dispatch",
    "value_dispatch",
]
