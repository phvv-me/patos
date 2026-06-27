# pipeline

A reversible stack of stages, applied in order on the way down and unwound in reverse on the way back.

## When to reach for it

Reach for `Pipeline` when a value passes through a chain of invertible transforms, something
happens at the bottom, and the chain has to unwind cleanly back to the original space. The
codec rotate, scale, round, decode, unrotate stack and the gauge chain are exactly this shape.

## The anti-pattern it replaces

It replaces the decorator that writes `transform.inverse(inner(transform.forward(x)))` by hand
in every method, and the nested forward/inverse calls that grow unreadable as the stack
deepens. Declare the stages once, and `apply` is the one-line "transform in, do the work,
transform out".

## Usage

```python
from patos import Pipeline


class Offset:
    """A reversible stage: add going forward, subtract coming back."""

    def __init__(self, by: int) -> None:
        self.by = by

    def forward(self, value: int) -> int:
        return value + self.by

    def inverse(self, value: int) -> int:
        return value - self.by


pipe = Pipeline((Offset(1), Offset(10)))
pipe.forward(0)              # 11 -- down through both stages
pipe.inverse(11)             # 0  -- back up in reverse order
pipe.apply(0, lambda x: x)   # 0  -- round-trips with an identity core
```

Each stage is any object with `forward`/`inverse`, the `Reversible` protocol.

## Public API

- `Pipeline(stages=())`. Build a pipeline from reversible stages, outermost first.
- `Pipeline.forward(value)`. Send the value down through every stage's `forward`.
- `Pipeline.inverse(value)`. Bring the value back up through every stage's `inverse`, in reverse.
- `Pipeline.apply(value, core=identity)`. Run `core` in the fully-transformed frame, then unwind.
- `Pipeline.then(stage)`. A new pipeline with `stage` appended as the new innermost stage.
- `Reversible`. The `forward`/`inverse` protocol a stage satisfies; runtime-checkable.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/pipeline.py"
```
