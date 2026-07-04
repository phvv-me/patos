from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reversible[Source, Frame](Protocol):
    """One invertible stage of a :class:`Pipeline`: a `forward` paired with its `inverse`.

    The contract the codec transforms (rotate, scale, whiten) and the gauge steps already meet:
    `forward` maps a value into the stage's frame and `inverse` brings it back, so wrapping a core
    operation in the stage and unwrapping after round-trips through the original space. Generic
    over the `Source` space and the in-stage `Frame` type, so a stage may change the type going
    forward (a pack/unpack) as long as `inverse` restores it. Both are invariant, since each
    appears once as an argument and once as a return across the two methods.
    """

    def forward(self, value: Source) -> Frame:
        """Map a value into this stage's frame."""

    def inverse(self, value: Frame) -> Source:
        """Bring a value in this stage's frame back to the original space."""


class Pipeline[T]:
    """A reversible stack of stages, applied in order forward and unwound in reverse.

    The shape the composed codec (rotate, scale, round, decode, unrotate) and the gauge chain
    both follow: send a value down through every stage's `forward`, then back up through every
    stage's `inverse` in the opposite order, so the stack is symmetric around whatever happens at
    the bottom. :meth:`forward` runs the descent, :meth:`inverse` the ascent, and :meth:`apply`
    wraps a core operation between them, the one-line spelling of "transform in, do the work,
    transform out" that a decorator like the codec `Composed` writes by hand per method.

    Stages compose left to right, so the first stage is the outermost frame: in a two-stage
    `Pipeline([rotate, scale])`, `forward` rotates then scales, and `inverse` unscales then
    unrotates. Each stage is any :class:`Reversible`, kept by reference, so a pipeline is a value
    object that adds no behaviour of its own beyond the ordering.

    stages: the reversible stages, outermost first.
    """

    def __init__(self, stages: tuple[Reversible[T, T], ...] = ()) -> None:
        self.stages = tuple(stages)

    def forward(self, value: T) -> T:
        """Send `value` down through every stage's `forward`, outermost stage first."""
        for stage in self.stages:
            value = stage.forward(value)
        return value

    def inverse(self, value: T) -> T:
        """Bring `value` back up through every stage's `inverse`, innermost stage first."""
        for stage in reversed(self.stages):
            value = stage.inverse(value)
        return value

    def apply(self, value: T, core: Callable[[T], T] = lambda value: value) -> T:
        """Run `core` in the fully-transformed frame, then unwind back to the original space.

        The round trip the stack exists for: `forward` the value down through every stage, apply
        `core` at the bottom (the round/decode the codec does there, identity by default so a bare
        `apply` is just `inverse(forward(value))`), then `inverse` back up. With reversible stages
        and an identity core this returns the input unchanged, which is the property a stage's
        `inverse` is defined by.

        value: the value entering the stack.
        core: the operation performed in the innermost frame.
        """
        return self.inverse(core(self.forward(value)))

    def then(self, stage: Reversible[T, T]) -> Pipeline[T]:
        """A new pipeline with `stage` appended as the new innermost stage."""
        return Pipeline((*self.stages, stage))

    def __len__(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:
        return f"<Pipeline stages={[type(stage).__name__ for stage in self.stages]}>"
