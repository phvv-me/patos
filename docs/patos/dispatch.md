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
- `.register(key=None, *, name=None)`. Return an implementation decorator. Empty parentheses infer the function name. A key or `name` sets it explicitly. Identifier keys are also exposed as attributes unless they would shadow the dispatcher's own API.
- `.registry`. Read-only view of the kind to implementation mapping.
- `.kinds()`. All registered kinds, sorted by `repr`.
- `key in dispatcher`. Membership test on registered kinds.

### type_dispatch

The dual that keys on the *type* of the first positional argument rather than a keyword value,
the open-type-ladder replacement for an `if isinstance(x, A): ... elif isinstance(x, B): ...`
chain. Like `functools.singledispatch` but with the same registration ergonomics as
`value_dispatch`.

```python
from patos import type_dispatch


@type_dispatch
def render(node):
    return "unknown"


@render.register()
def _(node: Circle):       # type read from the first parameter's annotation
    return "circle"


@render.register(Square)   # or given explicitly
def _(node):
    return "square"


render(Circle())           # "circle"
render(object())           # "unknown" -- unmatched type falls back
```

- `type_dispatch`. Decorator. Works bare (`@type_dispatch`) or parametrized (`@type_dispatch()`).
- `.register(type=None)`. Return an implementation decorator. Empty parentheses infer the first parameter's type annotation. An explicit type selects it directly. A call resolves the argument's type against the registry by walking its MRO, so the most specific registered base wins.
- `.registry`. Read-only view of the type to implementation mapping.
- `.types()`. All registered types, sorted by name.
- `cls in dispatcher`. Membership test on registered types.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/dispatch.py"
```
