from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from patos import Strategy, StrFlag, value_dispatch
from patos.flags import flags
from patos.strategy import StrategyError


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


flag_cases = [
    ({"verbose": True}, ("--verbose",)),
    ({"name": "duck"}, ("--name", "duck")),
    ({"count": 3}, ("--count", "3")),
    ({"mem_gb": 8}, ("--mem-gb", "8")),
    ({"skip": None}, ()),
    ({"skip": False}, ()),
    ({"skip": ""}, ()),
    ({"tags": []}, ()),
    ({"tags": ["a", "b"]}, ("--tags", "a", "--tags", "b")),
]


@pytest.mark.parametrize(("options", "expected"), flag_cases)
def test_flags_value_kinds(options: dict[str, object], expected: tuple[str, ...]) -> None:
    assert flags(**options) == expected


def test_flags_joined_and_separator() -> None:
    """joined emits --flag=value tokens; a separator collapses a sequence into one flag."""
    assert flags(name="duck", joined=True) == ("--name=duck",)
    assert flags(tags=["a", "b"], joined=True) == ("--tags=a", "--tags=b")
    assert flags(tags=["a", "b", "c"], separator=",") == ("--tags", "a,b,c")
    assert flags(tags=["a", "b"], separator=",", joined=True) == ("--tags=a,b",)


@given(value=st.integers() | st.text(min_size=1))
def test_flags_round_trips_scalar_value(value: int | str) -> None:
    """A non-empty scalar always renders as the flag followed by its stringified value."""
    assert flags(opt=value) == ("--opt", str(value))


@given(items=st.lists(st.text(min_size=1).filter(lambda s: "," not in s), min_size=1))
def test_flags_separator_collapses_to_one_token(items: Sequence[str]) -> None:
    """A separator renders a sequence as the flag and one joined value token."""
    argv = flags(items=items, separator=",")
    assert argv[0] == "--items"
    assert len(argv) == 2
    assert argv[1].split(",") == list(items)


def test_strflag_carries_string_and_or_combines() -> None:
    """Members keep their string with power-of-two values and OR-combine in declaration order."""
    class Opt(StrFlag):
        ALL = "-a"
        BIG = "--big"
        FAST = "--fast"

    assert Opt.ALL.string == "-a"
    assert [m.value for m in Opt] == [1, 2, 4]
    combined = Opt.ALL | Opt.FAST
    assert [m.string for m in combined] == ["-a", "--fast"]
