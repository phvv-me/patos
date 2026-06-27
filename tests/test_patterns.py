from typing import Self

import pytest
from hypothesis import given
from hypothesis import strategies as st

from patos import FlyweightMeta, Registry, Singleton
from patos.registry import CAMEL_BOUNDARY


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


def test_registry_implementations_exclude_pydantic_generic_aliases() -> None:
    """A concrete subclass that pins a generic base's type parameter adds no alias implementation.

    Subclassing a generic `Component` under PEP 695 makes pydantic materialize an intermediate
    parametrization class (`Combinator[Tensor]`) that enrolls in the registry under a bracketed
    name. Those aliases are typing artifacts of their origin, not separate providers, so
    `implementations`, `names` and `find` see only the genuine concrete subclass.
    """
    from patos import Component

    class Combinator[C](Component):
        inner: object

    class Wrap(Combinator[int]):
        pass

    impls = Combinator.implementations()
    assert Wrap in impls
    assert all("[" not in impl.name for impl in impls)
    assert "combinator[int]" not in Combinator.names()
    assert Combinator.find("wrap") is Wrap


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


def test_registry_implementations_are_scoped_to_the_calling_subtree() -> None:
    """Two sibling hierarchies sharing one root each see only their own implementations.

    The codec `Quantizer` and the `Lattice` oracle both enroll under one `Component` root, so the
    shared registry list holds both. `Lattice.implementations()` must still list only lattices and
    `Lattice.find` must scan only lattices, otherwise a quantizer's key (or a same-named class
    reachable through the root under a re-importing test runner) would collide on a key the lattice
    hierarchy never owns. The scope is the subtree of the calling class, not the whole root.
    """

    class Component(Registry):
        pass

    class Lattice(Component):
        pass

    class Quantizer(Component):
        pass

    class Leech(Lattice):
        name = "leech"

    class Trellis(Quantizer):
        name = "trellis"

    assert Lattice.root() is Quantizer.root() is Component
    assert Leech in Component.registry() and Trellis in Component.registry()
    assert Lattice.implementations() == [Leech]
    assert Quantizer.implementations() == [Trellis]
    assert Lattice.find("leech") is Leech
    with pytest.raises(KeyError):
        Lattice.find("trellis")  # the sibling quantizer's key is invisible to the lattice subtree


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


# Each word is two or more lowercase letters so its capitalised form is one capital over a
# lowercase run, never a single letter that would read as an acronym and fuse with its neighbour.
words = st.text("abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=6)


@given(parts=st.lists(words.map(str.capitalize), min_size=1, max_size=5))
def test_registry_kebab_derivation_round_trips(parts: list[str]) -> None:
    """A PascalCase join of lowercase words derives to those words joined by hyphens.

    Each segment starts with one capital over a lowercase run, so the only camel
    boundaries are between segments, giving a clean structural property: the derived
    key is exactly the lowercased words joined by `-`, with no stray, leading or
    trailing hyphens, and the class is findable under it.
    """
    pascal = "".join(parts)
    expected = "-".join(part.lower() for part in parts)

    class Base(Registry):
        pass

    impl = type(pascal, (Base,), {})

    assert impl.name == expected
    assert impl.name.strip("-") == impl.name
    assert "--" not in impl.name
    assert Base.find(expected) is impl
    assert set(Base.names()) == {expected}


def test_registry_kebab_keeps_acronym_with_digit_whole_and_round_trips() -> None:
    """A pure acronym stays one segment whether or not it carries a digit, and the key is stable.

    Before the fix the `[a-z0-9]` word-end class let a digit inside an acronym split off its
    trailing capital, so the real codec `E8P` derived to `e8-p` (and `RVQ`/`MOE` were only spared
    because they carry no digit). The key was then neither readable nor a fixed point: re-deriving
    its PascalCase form gave a different string. A capital that begins a lowercase *word* still
    splits, so `E8Lattice` keeps its boundary.
    """

    class Codec(Registry):
        pass

    class RVQ(Codec):
        pass

    class MOE(Codec):
        pass

    class E8P(Codec):
        pass

    class E8Lattice(Codec):
        pass

    assert (RVQ.name, MOE.name, E8P.name) == ("rvq", "moe", "e8p")
    assert E8Lattice.name == "e8-lattice"
    for impl in (RVQ, MOE, E8P, E8Lattice):
        assert Codec.find(impl.name) is impl
        # idempotency is the round-trip property: deriving an already-kebab name is a no-op
        rebuilt = "".join(part.capitalize() for part in impl.name.split("-"))
        assert CAMEL_BOUNDARY.sub("-", impl.name).lower() == impl.name
        assert CAMEL_BOUNDARY.sub("-", rebuilt).lower() == impl.name


def test_registry_select_filters_implementations_in_registration_order() -> None:
    """`select(predicate)` keeps the matching concrete impls, in registration order."""

    class Engine(Registry):
        capability = "base"

    class Evaluate(Engine):
        capability = "evaluate"

    class Factor(Engine):
        capability = "factor"

    class AlsoEvaluate(Engine):
        capability = "evaluate"

    evaluators = Engine.select(lambda impl: impl.capability == "evaluate")
    assert evaluators == [Evaluate, AlsoEvaluate]
    assert Engine.select(lambda impl: impl.capability == "factor") == [Factor]
    assert Engine.select(lambda impl: False) == []


def test_registry_first_available_picks_first_passing_probe() -> None:
    """`first_available` returns the first impl whose `is_available` classmethod is true."""

    class Backend(Registry):
        @classmethod
        def is_available(cls) -> bool:
            return False

    class Off(Backend):
        @classmethod
        def is_available(cls) -> bool:
            return False

    class On(Backend):
        @classmethod
        def is_available(cls) -> bool:
            return True

    class AlsoOn(Backend):
        @classmethod
        def is_available(cls) -> bool:
            return True

    assert Backend.first_available() is On
    assert Backend.first_available(lambda impl: impl is AlsoOn) is AlsoOn


def test_registry_first_available_treats_unprobed_impls_as_available() -> None:
    """An impl exposing no availability probe counts as available, so a plain last one wins."""

    class Backend(Registry):
        pass

    class Plain(Backend):
        pass

    assert Backend.first_available() is Plain


def test_registry_first_available_reads_available_method_and_raises_on_none() -> None:
    """The default probe also reads an `available` classmethod, and none available raises."""

    class Backend(Registry):
        pass

    class Never(Backend):
        @classmethod
        def available(cls) -> bool:
            return False

    with pytest.raises(LookupError, match="no available implementation"):
        Backend.first_available()


def test_registry_find_skips_impl_missing_the_custom_attribute() -> None:
    """`find` on a custom attribute ignores an impl that does not declare it on itself."""

    class Driver(Registry):
        pass

    class Http(Driver):
        scheme = "http"

    class Plain(Driver):
        pass  # no `scheme` of its own, so it never answers a scheme lookup

    assert Driver.find("http", attr="scheme") is Http
    assert Plain in Driver.implementations()
    with pytest.raises(KeyError) as miss:
        Driver.find("file", attr="scheme")
    assert "scheme='file'" in miss.value.args[0]


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


def test_flyweight_survives_uncomparable_args() -> None:
    """An argument whose ``bool(a == b)`` raises (a tensor) never crashes the cache lookup.

    A pydantic model carrying tensors reduces a tensor to a bool in its ``==`` and raises; two
    equal-but-distinct such arguments hash-collide, so a plain ``(type, value)`` key would raise
    resolving the collision. The flyweight keys by identity first, then a guarded compare, so it
    interns a reused object and degrades to per-object for an uncomparable one, never raising.
    """

    class Tensorish:
        def __init__(self, items: tuple[int, ...]) -> None:
            self.items = items

        def __hash__(self) -> int:
            return hash(self.items)  # equal items collide, forcing the lookup to compare

        def __eq__(self, other: object) -> bool:
            raise RuntimeError("ambiguous: equality is a vector, not a bool")

    class Node(metaclass=FlyweightMeta):
        def __init__(self, code: Tensorish) -> None:
            self.code = code

    shared = Tensorish((1, 2, 3))
    assert Node(shared) is Node(shared)  # the same object interns by identity, no compare
    first, second = Tensorish((4, 5)), Tensorish((4, 5))  # equal, distinct, hash-colliding
    assert Node(first) is not Node(second)  # uncomparable args degrade to per-object, no crash
    assert Node(first) is Node(first)  # each still interns by its own object


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


def test_registry_kebab_splits_embedded_acronyms() -> None:
    """An acronym run keeps its own segment instead of fusing into the next word.

    Before the fix `HTTPServer` derived to `httpserver` (no split at the acronym
    boundary), so an acronym-bearing class got an unreadable key that also collided
    with neither its camelCased twin nor `find`'s expectations.
    """

    class Codec(Registry):
        pass

    class HTTPServer(Codec):
        pass

    class XMLHttpRequest(Codec):
        pass

    class ParseURLToJSON(Codec):
        pass

    class IOError(Codec):
        pass

    assert HTTPServer.name == "http-server"
    assert XMLHttpRequest.name == "xml-http-request"
    assert ParseURLToJSON.name == "parse-url-to-json"
    assert IOError.name == "io-error"
    assert Codec.find("http-server") is HTTPServer


def test_registry_bare_annotation_does_not_suppress_derivation() -> None:
    """A bare `name: str` annotation (no value) still gets an auto-derived key.

    Before the fix the annotation suppressed derivation without assigning anything,
    so the subclass silently inherited the root's key and answered for the wrong name.
    """

    class Base(Registry):
        pass

    class Worker(Base):
        name: str

    assert Worker.name == "worker"
    assert Base.find("worker") is Worker
    assert "base" not in Base.names()
