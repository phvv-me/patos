from importlib.metadata import PackageNotFoundError, version

from .dispatch import value_dispatch
from .flyweight import FlyweightMeta
from .registry import Registry
from .singleton import Singleton, SingletonMeta
from .strategy import Available, Strategy, StrategyError
from .strflag import StrFlag

try:
    __version__ = version("patos")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Available",
    "FlyweightMeta",
    "Registry",
    "Singleton",
    "SingletonMeta",
    "StrFlag",
    "Strategy",
    "StrategyError",
    "value_dispatch",
]
