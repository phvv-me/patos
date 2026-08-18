import pytest

from patos import Resolution, Strategy, StrategyError


class _Link:
    def __init__(self, *, up: bool | None, boom: str | None = None) -> None:
        self.up = up
        self.boom = boom

    def available(self) -> bool:
        if self.boom is not None:
            raise ConnectionError(self.boom)
        return bool(self.up)


def test_cascade_returns_the_winner_with_the_rejection_log() -> None:
    transports: Strategy[_Link] = Strategy("transport")
    transports.register("ssh-22", _Link(up=None, boom="port 22 timed out"))
    transports.register("mosh", _Link(up=False))
    transports.register("tailnet", _Link(up=True))
    outcome = transports.cascade()
    assert isinstance(outcome, Resolution)
    assert outcome.winner == "tailnet"
    assert outcome.rejected == (
        ("ssh-22", "ConnectionError: port 22 timed out"),
        ("mosh", "reported unavailable"),
    )


def test_cascade_win_on_first_link_carries_no_rejections() -> None:
    transports: Strategy[_Link] = Strategy("transport")
    transports.register("ssh", _Link(up=True))
    transports.register("never-probed", _Link(up=None, boom="should not run"))
    outcome = transports.cascade()
    assert outcome.winner == "ssh" and outcome.rejected == ()


def test_cascade_all_rejected_names_every_reason() -> None:
    transports: Strategy[_Link] = Strategy("transport")
    transports.register("a", _Link(up=False))
    transports.register("b", _Link(up=None, boom="refused"))
    with pytest.raises(StrategyError, match="a: reported unavailable; b: ConnectionError: refused"):
        transports.cascade()


def test_cascade_treats_probeless_impls_as_available() -> None:
    plain: Strategy[str] = Strategy("plain")
    plain.register("fallback", "value")
    assert plain.cascade().implementation == "value"
