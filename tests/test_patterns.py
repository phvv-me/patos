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


def test_registry_names_lists_implementation_keys(
    codec_root: type[Registry],
    codec_impls: tuple[type[Registry], type[Registry]],
) -> None:
    """`names()` returns every concrete implementation's `name`, the keys `find` accepts."""
    assert sorted(codec_root.names()) == sorted(impl.name for impl in codec_impls)
    assert all(codec_root.find(name) in codec_impls for name in codec_root.names())


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


def test_registry_dispatch_tries_each_and_groups_all_refusals() -> None:
    """dispatch tries each impl, returns the first success, groups every refusal."""

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
    with pytest.raises(ExceptionGroup, match="refused dispatch") as failure:
        Codec.dispatch("nope")
    assert [str(error) for error in failure.value.exceptions] == ["not json", "yaml refuses"]


def test_registry_class_under_two_roots_enrolls_in_both() -> None:
    """A class deriving from two registry roots appears in both registries."""

    class Reader(Registry):
        pass

    class Writer(Registry):
        pass

    class File(Reader, Writer):
        pass

    assert File in Reader.registry()
    assert File in Writer.registry()
    assert File.root() is Reader
    assert Reader.registry() == [Reader, File]
    assert Writer.registry() == [Writer, File]


def test_registry_find_ignores_inherited_keys_and_rejects_duplicates() -> None:
    """find matches own attributes only and raises when two impls share a key."""

    class Codec(Registry):
        name = "base"

    class Json(Codec):
        name = "json"

    class Forgot(Json):
        pass

    assert Codec.find("json") is Json
    with pytest.raises(KeyError) as miss:
        Codec.find("base")
    assert "no implementation with name='base'" in miss.value.args[0]

    class Dup(Codec):
        name = "json"

    with pytest.raises(ValueError, match="duplicate name='json' on Json and Dup"):
        Codec.find("json")
    assert Forgot in Codec.registry()


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


def test_singleton_instance_lives_on_the_class_not_a_global() -> None:
    """The cached instance is stored per class, so no metaclass-level table pins it."""

    class Lone(Singleton):
        pass

    instance = Lone()
    assert Lone.__dict__["singleton_instance"] is instance
    assert "singleton_instance" not in Singleton.__dict__
    assert not hasattr(type(Singleton), "instances")


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


def test_flyweight_interning_is_typed() -> None:
    """Equal arguments of different types intern separately, like lru_cache(typed=True)."""

    class Node(metaclass=FlyweightMeta):
        def __init__(self, value: object = 0, *, tag: object = "x") -> None:
            self.value = value

    one, true, one_float = Node(1), Node(True), Node(1.0)
    assert one is not true
    assert one is not one_float
    assert true is not one_float
    assert Node(1) is one
    assert Node(tag=1) is not Node(tag=True)


def test_flyweight_caches_are_per_class() -> None:
    class A(metaclass=FlyweightMeta):
        pass

    class B(metaclass=FlyweightMeta):
        pass

    assert A() is A()
    assert A() is not B()


def test_registry_auto_derives_kebab_name_unless_declared() -> None:
    """`name` falls out of the class name, and an explicit declaration always wins."""

    class AudioCodec(Registry):
        pass

    class OpusFB8(AudioCodec):
        pass

    class FancyCodec(AudioCodec):
        name = "fancy"

    class Annotated(AudioCodec):
        name: str = "declared"

    assert AudioCodec.name == "audio-codec"
    assert OpusFB8.name == "opus-fb8"
    assert FancyCodec.name == "fancy"
    assert Annotated.name == "declared"
    assert AudioCodec.find("opus-fb8") is OpusFB8
