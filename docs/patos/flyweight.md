# flyweight

Intern instances by their constructor arguments, so equal arguments always yield the same object.

## When to reach for it

Reach for `flyweight` when many parts of your program ask for the same logical thing and you
want them to share one immutable instance. Colors, currencies, glyphs, units, parsed schemas,
anything that is value-like and expensive or wasteful to rebuild. The same arguments give you
the same object, built once and shared forever.

## The anti-pattern it replaces

It replaces `@cache` slapped over `__init__`, which does not actually work, since `__init__`
returns `None` and the cache caches nothing useful while the object is still rebuilt. It also
replaces ad-hoc interning dicts scattered across the codebase. `flyweight` moves interning into
the type, keeps instances immutable, and builds each one exactly once.

## Usage

`FlyweightMeta` is a metaclass. Set it on the class you want interned, and construction caches by
the constructor arguments.

```python
from patos import FlyweightMeta


class Color(metaclass=FlyweightMeta):
    def __init__(self, name: str) -> None:
        self.name = name


a = Color("teal")
b = Color("teal")
c = Color("amber")

assert a is b       # same args, same interned object
assert a is not c
```

Because `FlyweightMeta` composes with `ABCMeta`, a flyweight can also be abstract.

```python
from abc import abstractmethod
from patos import FlyweightMeta


class Shape(metaclass=FlyweightMeta):
    @abstractmethod
    def area(self) -> float: ...
```

## Public API

- `FlyweightMeta`. The metaclass that performs interning. Use it as `class X(metaclass=FlyweightMeta)`. The first construction with a given `(args, kwargs)` builds and caches the instance, and every later construction with equal arguments returns that same object without re-running `__init__`. Each class keeps its own cache and arguments must be hashable. It composes with `ABCMeta`, so a flyweight can also be an abstract base class. An argument whose equality is not a plain `bool` (a model carrying tensors, where `a == b` reduces a tensor and raises) is interned by identity rather than value, so the same object still shares one instance while two equal-but-distinct such arguments get separate instances rather than crashing the lookup.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/flyweight.py"
```
