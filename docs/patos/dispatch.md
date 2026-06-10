# dispatch

Dispatch on a value the way `functools.singledispatch` dispatches on a type.

## When to reach for it

Reach for `dispatch` when you select behavior by the value of an argument rather than its type.
Think `kind="html"` versus `kind="text"`, or a message routed by its `event` field. The dispatch
key is the value of a keyword argument, default `kind`, and each implementation registers under
the value it handles.

## The anti-pattern it replaces

It replaces the long `if kind == ...: elif kind == ...:` chain and the dict-of-functions you wire
up by hand. Implementations register with a decorator, the fallback handles the no-kind case, and
a clear error names the valid kinds when an unknown one arrives.

## Usage

```python
from patos import value_dispatch


@value_dispatch
def render(node: str, **kw: object) -> str:
    """Fallback used when no kind is given."""
    return node


@render.register("html")
def render_html(node: str, **kw: object) -> str:
    return f"<p>{node}</p>"


@render.register("text")
def render_text(node: str, **kw: object) -> str:
    return node.upper()


render("hi", kind="html")   # "<p>hi</p>"
render("hi", kind="text")   # "HI"
render("hi")                # "hi", the fallback
render.html("hi")           # "<p>hi</p>", direct attribute call
```

Use the parametrized form to dispatch on a different keyword.

```python
@value_dispatch(kind="how")
def emit(node: str, **kw: object) -> str: ...
```

## Public API

- `value_dispatch`. Decorator. Works bare (`@value_dispatch`) or parametrized (`@value_dispatch(kind="how")`).
- `.register(key=None, *, name=None)`. Register an implementation, bare for the function name, or with an explicit key. Identifier keys are also exposed as attributes unless they would shadow the dispatcher's own API.
- `.registry`. Read-only view of the kind to implementation mapping.
- `.kinds()`. All registered kinds, sorted by `repr`.
- `key in dispatcher`. Membership test on registered kinds.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/dispatch.py"
```
