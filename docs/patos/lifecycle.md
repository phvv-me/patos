# lifecycle

A typed state machine that permits only the transitions you declare, and raises on any other.

## When to reach for it

Reach for `Lifecycle` when an object moves through a fixed set of states and only some moves
between them make sense. A job goes from running to one of ok, failed, pruned or crashed, never
back to running. Declare the allowed edges once, and the machine enforces them.

## The anti-pattern it replaces

It replaces the ad-hoc status string every writer is trusted to set sensibly, where nothing
stops a finished job sliding back to running or a pruned trial reporting success. The transition
table makes the legal moves explicit and an illegal one loud.

## Usage

```python
from patos import Lifecycle, IllegalTransition


job = Lifecycle(
    {
        "running": {"ok", "failed", "pruned", "crashed"},
        "ok": set(),
        "failed": set(),
    },
    initial="running",
)

job.to("ok")          # advances; returns "ok"
job.current           # "ok"
job.is_terminal()     # True -- no edge leaves "ok"
job.to("running")     # raises IllegalTransition naming the allowed moves
```

It is generic over the state type, so enum members work exactly like strings.

## Public API

- `Lifecycle(transitions, initial)`. Build the machine from a state to reachable-states table.
- `Lifecycle.current`. The latest state.
- `Lifecycle.allowed(target)`. Whether a move to `target` is a declared edge.
- `Lifecycle.to(target)`. Advance to `target` if allowed, else raise `IllegalTransition`.
- `Lifecycle.is_terminal()`. Whether no transition leaves the current state.
- `IllegalTransition`. Raised on a move the table does not allow.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/lifecycle.py"
```
