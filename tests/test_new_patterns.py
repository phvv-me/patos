import pytest

from patos import (
    Decorator,
    DerivedCache,
    IllegalTransition,
    Lifecycle,
    Pipeline,
    Reversible,
    type_dispatch,
)


class Offset:
    """A reversible stage that adds a constant going forward and subtracts it coming back."""

    def __init__(self, by: int) -> None:
        self.by = by

    def forward(self, value: int) -> int:
        return value + self.by

    def inverse(self, value: int) -> int:
        return value - self.by


def test_pipeline_forward_descends_and_inverse_ascends_in_opposite_order() -> None:
    """`forward` runs stages left to right; `inverse` unwinds them right to left."""
    pipeline = Pipeline((Offset(1), Offset(10)))

    assert pipeline.forward(0) == 11
    assert pipeline.inverse(11) == 0
    assert len(pipeline) == 2
    assert "Offset" in repr(pipeline)


def test_pipeline_apply_round_trips_through_a_core_operation() -> None:
    """`apply` transforms in, runs the core at the bottom, transforms back out."""
    pipeline = Pipeline((Offset(2), Offset(3)))

    # core triples the value seen in the fully-transformed frame (0 -> 5 -> 15), then unwinds
    assert pipeline.apply(0, lambda x: x * 3) == 15 - 3 - 2
    # the defining property: reversible stages with an identity core return the input unchanged
    assert pipeline.apply(7) == 7


def test_pipeline_then_appends_an_innermost_stage_without_mutating_the_original() -> None:
    """`then` returns a new pipeline with the stage appended; the original is untouched."""
    base = Pipeline((Offset(1),))
    extended = base.then(Offset(100))

    assert extended.forward(0) == 101
    assert base.forward(0) == 1
    assert len(base) == 1


def test_pipeline_empty_is_the_identity() -> None:
    """A pipeline with no stages forwards, inverses and applies as the identity."""
    pipeline: Pipeline[int] = Pipeline()

    assert pipeline.forward(5) == 5
    assert pipeline.inverse(5) == 5
    assert pipeline.apply(5) == 5


def test_offset_satisfies_the_reversible_protocol() -> None:
    """A plain forward/inverse pair is a `Reversible` at runtime, no base class needed."""
    assert isinstance(Offset(1), Reversible)
    assert not isinstance(object(), Reversible)


def test_decorator_forwards_everything_but_the_overrides() -> None:
    """A subclass overrides only what differs; every other access reads through to `wrapped`."""

    class Inner:
        def __init__(self) -> None:
            self.size = 8

        def label(self) -> str:
            return "inner"

        def kind(self) -> str:
            return "raw"

    class Loud(Decorator[Inner]):
        def label(self) -> str:
            return self.wrapped.label().upper()

    loud = Loud(Inner())
    assert loud.label() == "INNER"  # overridden
    # `Decorator.__getattr__` forwards to `wrapped` for whatever attribute is missing, so its
    # return type is honestly `object` (there is no way to know ahead of the call whether a
    # forwarded name is data or a method); calling or reading it back is exactly what the
    # class's docstring promises, and irreducible to anything narrower without per-attribute
    # overloads on an open-ended `wrapped`.
    assert loud.kind() == "raw"  # pyrefly: ignore[not-callable]  # delegated method
    assert loud.size == 8  # delegated data
    assert "Loud wrapping" in repr(loud)


def test_decorator_writes_land_on_the_decorator_not_the_wrapped_object() -> None:
    """Setting an attribute on the decorator shadows the wrapped one and leaves it untouched."""

    class Inner:
        def __init__(self) -> None:
            self.size = 8

    inner = Inner()
    decorator = Decorator(inner)
    # `Decorator` declares no fixed attribute set on purpose: writes are meant to land on
    # whichever instance they are made on (the decorator here, never `wrapped`), so `size` is
    # a new attribute rather than one type-checkable ahead of the assignment.
    decorator.size = 16  # pyrefly: ignore[missing-attribute]

    assert decorator.size == 16  # pyrefly: ignore[missing-attribute]
    assert inner.size == 8


def test_decorator_missing_attribute_raises_attribute_error() -> None:
    """An attribute on neither the decorator nor the wrapped object raises AttributeError."""
    decorator = Decorator(object())
    with pytest.raises(AttributeError, match="absent"):
        _ = decorator.absent


def test_decorator_access_before_wrapped_is_set_does_not_recurse() -> None:
    """Delegating before `wrapped` is assigned names the attribute instead of recursing forever."""

    class Early(Decorator[object]):
        def __init__(self) -> None:
            # touch a delegated attribute before super().__init__ sets `wrapped`
            self.tag = self.flavour  # type: ignore[attr-defined]

    with pytest.raises(AttributeError, match="no wrapped object yet"):
        Early()


def test_lifecycle_allows_declared_transitions_and_blocks_the_rest() -> None:
    """A declared edge advances the state; an undeclared one raises naming the allowed moves."""
    transitions = {
        "running": {"ok", "failed", "pruned", "crashed"},
        "ok": set(),
        "failed": set(),
    }
    job: Lifecycle[str] = Lifecycle(transitions, "running")

    assert job.allowed("ok")
    assert job.to("ok") == "ok"
    assert job.current == "ok"
    assert job.is_terminal()

    with pytest.raises(IllegalTransition, match="cannot move from 'ok' to 'running'"):
        job.to("running")


def test_lifecycle_rejects_an_undeclared_target_from_a_live_state() -> None:
    """A target absent from the current state's edge set is refused even mid-run."""
    job: Lifecycle[str] = Lifecycle({"running": {"ok", "failed"}}, "running")

    with pytest.raises(IllegalTransition, match="allowed from 'running'"):
        job.to("vanished")
    assert job.current == "running"


def test_lifecycle_state_absent_from_the_table_is_terminal() -> None:
    """A state with no entry in the table has no outgoing edge, so it reads as terminal."""
    job: Lifecycle[str] = Lifecycle({"running": {"done"}}, "running")
    job.to("done")

    assert job.is_terminal()
    assert "terminal" in repr(job)
    with pytest.raises(IllegalTransition):
        job.to("running")


def test_lifecycle_works_over_enum_states() -> None:
    """The machine is generic, so enum members transition exactly like strings."""
    from enum import Enum

    class State(Enum):
        START = "start"
        END = "end"

    flow: Lifecycle[State] = Lifecycle({State.START: {State.END}}, State.START)
    assert flow.to(State.END) is State.END
    assert flow.is_terminal()


def test_derived_cache_builds_once_per_key_then_reuses() -> None:
    """`get` runs `build` on the first miss for a key and returns the cached value after."""
    calls: list[tuple[str, int]] = []

    def build(role: str, rank: int) -> str:
        calls.append((role, rank))
        return f"{role}-{rank}"

    cache: DerivedCache[tuple[str, int], str] = DerivedCache()

    assert cache.get(("q", 4), lambda: build("q", 4)) == "q-4"
    assert cache.get(("q", 4), lambda: build("q", 4)) == "q-4"  # cache hit, build skipped
    assert cache.get(("k", 8), lambda: build("k", 8)) == "k-8"

    assert calls == [("q", 4), ("k", 8)]
    assert len(cache) == 2
    assert ("q", 4) in cache
    assert ("v", 2) not in cache
    assert "size=2" in repr(cache)


def test_type_dispatch_routes_on_the_first_argument_type() -> None:
    """Each registration form dispatches on `args[0]`'s type; an unmatched type falls back."""

    class Shape:
        pass

    class Circle(Shape):
        pass

    class Square(Shape):
        pass

    @type_dispatch
    def render(node: object = None) -> str:
        return "unknown"

    @render.register
    def _(node: Circle) -> str:
        return "circle"

    @render.register(Square)
    def _(node: Square) -> str:
        return "square"

    @render.register()
    def _(node: Shape) -> str:  # empty-parens form returns a decorator, type from the annotation
        return "shape"

    assert render(Circle()) == "circle"
    assert render(Square()) == "square"
    assert render(Shape()) == "shape"  # the Shape base registration now answers
    assert render() == "unknown"  # no positional argument, routes to the fallback
    assert Circle in render
    assert render.types() == [Circle, Shape, Square]
    assert set(render.registry) == {Circle, Square, Shape}
    assert "render" in repr(render)


def test_type_dispatch_resolves_to_the_nearest_registered_base() -> None:
    """A subtype with no own registration falls through to a registered supertype via the MRO."""

    class Animal:
        pass

    class Dog(Animal):
        pass

    class Puppy(Dog):
        pass

    @type_dispatch
    def speak(node: object) -> str:
        return "..."

    @speak.register(Animal)
    def _(node: Animal) -> str:
        return "animal"

    @speak.register(Dog)
    def _(node: Dog) -> str:
        return "woof"

    assert speak(Puppy()) == "woof"  # nearest base is Dog, not Animal
    assert speak(Dog()) == "woof"
    assert speak(Animal()) == "animal"


def test_type_dispatch_parametrised_form_and_unbound_errors() -> None:
    """The bare-decorator stage binds the fallback; calling before binding explains the gap."""
    dispatcher: type_dispatch[..., str] = type_dispatch()
    assert "unbound" in repr(dispatcher)
    with pytest.raises(TypeError, match="no function bound yet"):
        dispatcher("x")
    with pytest.raises(TypeError, match="no function bound yet"):
        dispatcher()

    @dispatcher
    def render(node: object) -> str:
        return "default"

    assert render is dispatcher
    assert dispatcher(object()) == "default"


def test_type_dispatch_rejects_method_fallbacks() -> None:
    """A fallback whose first parameter is self is refused with guidance, like value_dispatch."""
    with pytest.raises(TypeError, match="module-level function"):

        @type_dispatch
        def method(self: object) -> str:
            return "nope"


def test_type_dispatch_bare_register_needs_a_class_annotation() -> None:
    """Bare `register` reads the first parameter's annotation, rejecting a missing/non-class."""

    @type_dispatch
    def render(node: object) -> str:
        return "default"

    with pytest.raises(TypeError, match="no annotation"):

        @render.register
        def _(node) -> str:  # type: ignore[no-untyped-def]
            return "x"

    with pytest.raises(TypeError, match="not\n? *a class"):

        @render.register
        def _(node: int | str) -> str:
            return "x"
