import pytest

from patos import FieldProjection, Projection


class Symbolic(Projection[str]):
    """Project each field to a symbol tagged with its name."""

    x: int
    label: str = "none"

    @classmethod
    def __project__(cls, name: str) -> str:
        return f"sym({name})"


class Bare(Projection[str]):
    value: int = 0


def test_class_access_projects_and_instance_access_returns_values() -> None:
    """Class attribute access projects while instances stay ordinary pydantic."""
    assert Symbolic.x == "sym(x)"
    assert Symbolic.label == "sym(label)"
    point = Symbolic(x=3)
    assert point.x == 3
    assert point.label == "none"
    assert point.model_dump() == {"x": 3, "label": "none"}


def test_subclass_inherits_and_extends_projection() -> None:
    """A subclass projects inherited and new fields through the same hook."""

    class Extended(Symbolic):
        y: float = 0.0

    assert Extended.y == "sym(y)"
    assert Extended.x == "sym(x)"
    assert Extended(x=1, y=2.5).y == 2.5


def test_default_hook_refuses_projection() -> None:
    """Without an implemented hook, class access fails loudly."""
    with pytest.raises(NotImplementedError, match="value"):
        Bare.value  # noqa: B018
    assert Bare(value=7).value == 7


def test_descriptor_rejects_direct_instance_access() -> None:
    descriptor = FieldProjection[str]("x")

    with pytest.raises(AttributeError, match="x"):
        descriptor.__get__(Symbolic(x=3), Symbolic)
