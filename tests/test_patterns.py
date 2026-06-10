from typing import Self

import pytest
from hypothesis import given
from hypothesis import strategies as st

from patos import FlyweightMeta, Registry, Singleton


def test_registry_roots_and_membership() -> None:
    """A root owns itself and its subclasses in one shared list; no root means no lookup."""

    class Shape(Registry):
        pass

    class Circle(Shape):
        pass

    class Square(Shape):
        pass

    assert Shape.is_registry_root()
    assert not Circle.is_registry_root()
    assert Shape.registry() == [Shape, Circle, Square]
    assert Circle.registry() is Shape.registry()
    with pytest.raises(RuntimeError, match="No registry"):
        Registry.registry()


def test_registry_implementations_excludes_root_and_abstract_bases(
    codec_root: type[Registry],
    codec_impls: tuple[type[Registry], type[Registry]],
) -> None:
    """`implementations()` keeps only concrete members, dropping the root and abstract bases."""
    impls = codec_root.implementations()

    assert tuple(impls) == codec_impls
    assert codec_root not in impls
    assert codec_root.root() is codec_root
    assert all(not c.__abstractmethods__ for c in impls)
    assert set(impls) < set(codec_root.registry())


def test_registry_find_by_name_and_clear_error_on_miss(
    codec_root: type[Registry],
    codec_impls: tuple[type[Registry], type[Registry]],
) -> None:
    """`find` resolves a concrete impl by its `name` and reports the known keys on a miss."""
    json_impl, yaml_impl = codec_impls

    assert codec_root.find("json") is json_impl
    assert codec_root.find("yaml") is yaml_impl
    with pytest.raises(KeyError) as miss:
        codec_root.find("binary")
    message = miss.value.args[0]
    assert "no implementation with name='binary'" in message
    assert "'json'" in message and "'yaml'" in message
    assert "'base'" not in message and "'binary'" not in message.split("Known")[1]


def test_registry_find_honors_custom_attribute() -> None:
    """`find` can key on any class attribute, not just the default `name`."""

    class Driver(Registry):
        scheme = "base"

    class Http(Driver):
        scheme = "http"

    class File(Driver):
        scheme = "file"

    assert Driver.find("http", attr="scheme") is Http
    assert Driver.find("file", attr="scheme") is File
    with pytest.raises(KeyError) as miss:
        Driver.find("ftp", attr="scheme")
    assert "scheme='ftp'" in miss.value.args[0]


def test_registry_dispatch_tries_each_and_reraises_last() -> None:
    """dispatch tries each impl, returns the first success, re-raises the last error."""

    class Codec(Registry):
        @classmethod
        def from_dispatch(cls, *args: object, **kwargs: object) -> Self:
            raise ValueError("base")

    class Json(Codec):
        @classmethod
        def from_dispatch(cls, raw: str) -> Self:
            if raw != "json":
                raise ValueError("not json")
            return cls()

    class Yaml(Codec):
        @classmethod
        def from_dispatch(cls, raw: str) -> Self:
            raise TypeError("yaml refuses")

    assert isinstance(Codec.dispatch("json"), Json)
    assert isinstance(Json.dispatch("json"), Json)
    with pytest.raises(TypeError, match="yaml refuses"):
        Codec.dispatch("nope")


def test_registry_dispatch_with_no_impls_raises_runtime() -> None:
    class Empty(Registry):
        pass

    with pytest.raises(RuntimeError, match="No implementation"):
        Empty.dispatch()


def test_registry_default_from_dispatch_not_implemented() -> None:
    class Plain(Registry):
        pass

    with pytest.raises(NotImplementedError, match="from_dispatch"):
        Plain.from_dispatch()


def test_singleton_one_instance_init_runs_once() -> None:
    class Counter(Singleton):
        def __init__(self) -> None:
            self.created = getattr(self, "created", 0) + 1

    first, second = Counter(), Counter()
    assert first is second
    assert first.created == 1

    class Other(Singleton):
        pass

    assert Other() is not first


@given(
    a=st.integers() | st.text(),
    b=st.integers() | st.text(),
)
def test_flyweight_interns_by_args(a: int | str, b: int | str) -> None:
    """Equal args share one instance without re-running __init__; distinct args share iff equal."""

    class Node(metaclass=FlyweightMeta):
        def __init__(self, value: int | str, *, tag: str = "x") -> None:
            self.value = value
            self.built = getattr(self, "built", 0) + 1

    same_args = Node(a) is Node(a)
    by_kwargs = Node(a, tag="k") is Node(a, tag="k")
    distinct = Node(a) is Node(b)

    assert same_args
    assert by_kwargs
    assert distinct == (type(a) is type(b) and a == b)
    assert Node(a).built == 1


def test_flyweight_caches_are_per_class() -> None:
    class A(metaclass=FlyweightMeta):
        pass

    class B(metaclass=FlyweightMeta):
        pass

    assert A() is A()
    assert A() is not B()
