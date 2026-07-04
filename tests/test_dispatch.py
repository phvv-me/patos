import functools

import pytest

from patos import Strategy, StrategyError, StrFlag, value_dispatch


def test_strategy_select_resolves_default_and_errors() -> None:
    """Registrations resolve, fall back to a default, and error cleanly."""
    s: Strategy[str] = Strategy("codecs")
    s.register("eager", "E")

    @s.add("lazy")
    def build_lazy() -> str:
        return "L"

    s.factory("named", lambda: "N")

    assert s.select("eager") == "E"
    assert s.select("lazy") == "L"
    assert s.select("named") == "N"
    assert s.select("missing", default="eager") == "E"
    assert s.names == ("eager", "lazy", "named")
    assert list(s) == ["eager", "lazy", "named"]
    assert "eager" in s and len(s) == 3
    assert "codecs" in repr(s)
    with pytest.raises(StrategyError):
        s.select("missing")
    with pytest.raises(StrategyError):
        s.select("missing", default="also-missing")


def test_strategy_error_is_a_lookup_error_with_clean_rendering() -> None:
    """StrategyError renders its message verbatim and still reads as a lookup failure."""
    s: Strategy[str] = Strategy("codecs")
    with pytest.raises(StrategyError) as failure:
        s.select("missing")
    assert isinstance(failure.value, LookupError)
    assert str(failure.value).startswith("codecs: no implementation")


def test_strategy_factory_rebind_invalidates_cache() -> None:
    """Re-registering a factory after resolution serves the new build, not the stale one."""
    s: Strategy[str] = Strategy()
    s.factory("codec", lambda: "old")
    assert s.select("codec") == "old"

    s.factory("codec", lambda: "new")
    assert s.select("codec") == "new"


def test_strategy_first_available_picks_first_true() -> None:
    """first_available picks the first true available(), treating plain impls as available."""

    class Maybe:
        def __init__(self, ok: bool) -> None:
            self.ok = ok

        def available(self) -> bool:
            return self.ok

    s: Strategy[object] = Strategy()
    s.register("no", Maybe(False))
    winner = Maybe(True)
    s.register("yes", winner)
    s.register("plain", object())

    assert s.first_available() is winner

    empty: Strategy[object] = Strategy()
    empty.register("never", Maybe(False))
    with pytest.raises(StrategyError):
        empty.first_available()

    plain_only: Strategy[object] = Strategy()
    sentinel = object()
    plain_only.register("plain", sentinel)
    assert plain_only.first_available() is sentinel


def test_strategy_first_available_honors_boolean_attribute() -> None:
    """A plain `available` data attribute counts as the truth value, not as callable."""

    class Flagged:
        def __init__(self, available: bool) -> None:
            self.available = available

    s: Strategy[Flagged] = Strategy()
    s.register("off", Flagged(False))
    winner = Flagged(True)
    s.register("on", winner)

    assert s.first_available() is winner


def sample(node: object, **_: object) -> str:
    return "fallback"


def test_value_dispatch_registration_forms_and_dispatch() -> None:
    """Every registration form dispatches, kind=None falls back, an unknown kind raises."""
    render = value_dispatch(sample)

    @render.register
    def html(node: object) -> str:
        return "html"

    @render.register("xml")
    def to_xml(node: object) -> str:
        return "xml"

    @render.register(name="json")
    def to_json(node: object) -> str:
        return "json"

    def existing(node: object) -> str:
        return "csv"

    render.register(existing, name="csv")

    assert render(None, kind="html") == "html"
    assert render.html(None) == "html"
    assert render(None, kind="xml") == "xml"
    assert render(None, kind="json") == "json"
    assert render(None, kind="csv") == "csv"
    assert render(None) == "fallback"
    assert render(None, kind=None) == "fallback"
    assert set(render.registry) == {"html", "xml", "json", "csv"}
    assert render.kinds() == sorted(["html", "xml", "json", "csv"], key=repr)
    assert "html" in render
    assert "sample" in repr(render)
    with pytest.raises(ValueError, match="unknown kind"):
        render(None, kind="nope")


def test_value_dispatch_parametrised_form_and_unbound_repr() -> None:
    """Reports unbound before binding; non-identifier keys dispatch without an attribute."""
    dispatcher = value_dispatch(kind="how")
    assert "unbound" in repr(dispatcher)

    @dispatcher
    def draw(node: object, **_: object) -> str:
        return "default"

    @dispatcher.register("9bad")
    def weird(node: object) -> str:
        return "weird"

    assert draw is dispatcher
    assert dispatcher(None, how="9bad") == "weird"
    assert not hasattr(dispatcher, "9bad")


def test_value_dispatch_impl_named_after_api_does_not_clobber_it() -> None:
    """Keys shadowing dispatcher API dispatch by key while the API keeps working."""
    render = value_dispatch(sample)

    @render.register
    def register(node: object) -> str:
        return "registered"

    @render.register
    def kinds(node: object) -> str:
        return "kinds"

    assert render(None, kind="register") == "registered"
    assert render(None, kind="kinds") == "kinds"
    assert callable(render.register)
    assert render.kinds() == ["kinds", "register"]

    @render.register
    def html(node: object) -> str:
        return "html"

    assert render(None, kind="html") == "html"
    assert render.html(None) == "html"

    @render.register("html")
    def html_two(node: object) -> str:
        return "html2"

    assert render.html(None) == "html2"


def test_value_dispatch_callable_keys_and_nameless_impl_error() -> None:
    """A class or partial works as a dispatch key; a nameless bare impl errors clearly."""

    class Markdown:
        pass

    render = value_dispatch(sample)

    @render.register(Markdown)
    def markdown(node: object) -> str:
        return "md"

    assert render(None, kind=Markdown) == "md"
    assert Markdown in render
    assert callable(Markdown)

    nameless = functools.partial(sample, None)
    with pytest.raises(TypeError, match="no __name__"):
        render.register()(nameless)
    render.register(name="partial")(nameless)
    # `__call__` types its arguments against the fallback's own `P` (`sample`'s `node`), but a
    # caller who knows the registered impl is a zero-arg `functools.partial` may skip what that
    # impl no longer needs; the dispatcher forwards whatever it is actually given rather than
    # reconstructing `P` from thin air, so this is a real, intentionally off-contract call.
    assert render(kind="partial") == "fallback"  # pyrefly: ignore[missing-argument]


def test_value_dispatch_fallback_dict_does_not_overwrite_state() -> None:
    """update_wrapper must not copy the fallback's __dict__ over dispatcher internals."""

    def messy(node: object, **_: object) -> str:
        return "messy"

    messy.kind_arg = "boom"  # type: ignore[attr-defined]
    messy.registry_map = "boom"  # type: ignore[attr-defined]

    dispatcher = value_dispatch(messy)
    assert dispatcher.kind_arg == "kind"
    assert dispatcher.registry_map == {}
    assert dispatcher(None) == "messy"


def test_value_dispatch_unbound_call_raises_clear_type_error() -> None:
    """Calling a parametrised dispatcher before binding explains what is missing."""
    dispatcher = value_dispatch(kind="how")
    with pytest.raises(TypeError, match="no function bound yet"):
        dispatcher(None, how="x")
    with pytest.raises(TypeError, match="no function bound yet"):
        dispatcher()


def test_value_dispatch_rejects_method_fallbacks() -> None:
    """A fallback whose first parameter is self is refused with guidance."""
    with pytest.raises(TypeError, match="module-level function"):

        @value_dispatch
        def method(self: object, **_: object) -> str:
            return "nope"


def test_strflag_carries_string_and_or_combines() -> None:
    """Members keep their string with power-of-two values and OR-combine in declaration order."""

    class Opt(StrFlag):
        ALL = "-a"
        BIG = "--big"
        FAST = "--fast"

    assert Opt.ALL.string == "-a"
    assert Opt.ALL.literal == "-a"
    assert [m.value for m in Opt] == [1, 2, 4]
    combined = Opt.ALL | Opt.FAST
    assert [m.string for m in combined] == ["-a", "--fast"]


def test_strflag_composite_and_empty_members_have_string() -> None:
    """Composite and empty members expose `.string` by joining decomposed literals."""

    class Opt(StrFlag):
        ALL = "-a"
        BIG = "--big"

    assert (Opt.ALL | Opt.BIG).string == "-a --big"
    assert (Opt.ALL & Opt.BIG).string == ""
    assert Opt(0).string == ""
