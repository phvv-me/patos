from typing import Self

import pytest
from hypothesis import given
from hypothesis import strategies as st

from patos import Flyweight, Registry, Singleton


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
    class Node(Flyweight):
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
    class A(Flyweight):
        pass

    class B(Flyweight):
        pass

    assert A() is A()
    assert A() is not B()
