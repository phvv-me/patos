from __future__ import annotations

from typing import Generic, TypeVar

S = TypeVar("S")


class IllegalTransition(RuntimeError):
    """Raised when a :class:`Lifecycle` is asked to move along an edge its table does not allow."""


class Lifecycle(Generic[S]):
    """A typed state machine that permits only the transitions its table declares.

    The replacement for the ad-hoc status string a long-running unit carries, where every writer
    is trusted to set a sensible next value and nothing stops a finished job sliding back to
    running or a pruned trial reporting success. Declare the allowed edges once as a table mapping
    each state to the set it may move to, and :meth:`to` enforces them: a legal move advances the
    current state, an illegal one raises :class:`IllegalTransition` naming the offending edge and
    the moves that were allowed.

    A state absent from the table (or mapped to an empty set) is terminal, so any move off it
    fails. The machine is generic over the state type, so strings and enum members work the same,
    and `current` always reads back the latest state.

    transitions: each state mapped to the states reachable from it in one step.
    initial: the state the machine starts in.
    """

    def __init__(self, transitions: dict[S, set[S]], initial: S) -> None:
        self.transitions = transitions
        self.current = initial

    def allowed(self, target: S) -> bool:
        """Whether a move from the current state to `target` is one of the declared edges."""
        return target in self.transitions.get(self.current, set())

    def to(self, target: S) -> S:
        """Advance to `target` if the edge is allowed, else raise :class:`IllegalTransition`.

        target: the state to move to; returned on success so a caller can chain the new state.
        """
        if not self.allowed(target):
            reachable = sorted(map(repr, self.transitions.get(self.current, set())))
            raise IllegalTransition(
                f"cannot move from {self.current!r} to {target!r}; "
                f"allowed from {self.current!r}: {reachable}.",
            )
        self.current = target
        return target

    def is_terminal(self) -> bool:
        """Whether no transition leaves the current state, so the lifecycle has settled."""
        return not self.transitions.get(self.current, set())

    def __repr__(self) -> str:
        marker = " (terminal)" if self.is_terminal() else ""
        return f"<Lifecycle at {self.current!r}{marker}>"
