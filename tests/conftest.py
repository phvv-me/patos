from abc import ABC, abstractmethod

import pytest

from patos import Registry


class Codec(Registry, ABC):
    """An abstract registry root with a keyed, concrete subclass family.

    `Codec` itself is the root and `Binary` is an abstract intermediate base, so both must be
    excluded from `implementations()`; `Json` and `Yaml` are the only concrete keyed members.
    """

    name = "base"

    @abstractmethod
    def extension(self) -> str: ...


class Binary(Codec, ABC):
    """Abstract intermediate base: still carries `extension`, so it stays abstract."""

    name = "binary"


class Json(Codec):
    name = "json"

    def extension(self) -> str:
        return ".json"


class Yaml(Codec):
    name = "yaml"

    def extension(self) -> str:
        return ".yaml"


@pytest.fixture
def codec_root() -> type[Codec]:
    """The abstract `Codec` registry root with `Json`/`Yaml` as its concrete implementations."""
    return Codec


@pytest.fixture
def codec_impls() -> tuple[type[Codec], type[Codec]]:
    """The concrete codec implementations, in registration order."""
    return (Json, Yaml)
