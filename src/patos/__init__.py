from importlib.metadata import PackageNotFoundError, version

from .dispatch import value_dispatch
from .flags import flags
from .flyweight import Flyweight, FlyweightMeta
from .registry import Registry
from .singleton import Singleton
from .strategy import Strategy
from .strflag import StrFlag

try:
    __version__ = version("patos")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Flyweight",
    "FlyweightMeta",
    "Registry",
    "Singleton",
    "StrFlag",
    "Strategy",
    "flags",
    "value_dispatch",
]
