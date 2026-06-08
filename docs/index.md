<div class="hero" markdown>

![patos logo](assets/logo.svg){ .hero-logo }

# patos

<p class="tagline">A flock of typed Python patterns. Pip one in, or copy one in and own the file.</p>

</div>

Each pattern is a single self-contained `pato` (a duck). Adopt one, and you get a tiny,
fully typed, zero-dependency module that does one thing well. The flock is `patos`. A single
duck is a `pato`.

## Two ways to use a pato

<span class="pill">pip it in</span> Install the whole flock and import the patterns you want.

```sh
pip install patos
```

```python
from patos import Registry, Singleton
```

<span class="pill">copy it in</span> Open the duck's page, copy its source straight from the docs, and paste it
into your own source tree. No runtime dependency, no version pin, no tool, just one module you can
read and change.

Every pato page ends with a **Source** section that shows the full module with a one-click copy
button. The file is yours the moment you paste it. Edit it, rename it, delete the parts you do not
need.

## Why patos

- **Typed.** Every pato ships full type hints and `py.typed`, so your checker sees through it.
- **Zero dependencies.** Each duck is standard library only. Copying one in adds nothing to your lockfile.
- **You own the code.** The copy-in path gives you a small, readable module instead of a black box.
- **One idea per duck.** No god-objects, no framework. Reach for the one pattern you need.

## The flock

<div class="grid cards" markdown>

-   ### [registry](patos/registry.md)

    Self-registering class hierarchies. Subclasses enroll through `__init_subclass__`, and
    `dispatch` tries each one until it builds.

-   ### [singleton](patos/singleton.md)

    One instance per class, with `__init__` running exactly once. Metaclass based, never a
    cached `__new__`.

-   ### [flyweight](patos/flyweight.md)

    Intern instances by their constructor args. Build once, share forever, stay immutable.

-   ### [strategy](patos/strategy.md)

    A named family of interchangeable implementations, picked at runtime or by first available.

-   ### [dispatch](patos/dispatch.md)

    Dispatch on a value the way `singledispatch` dispatches on a type.

-   ### [flags](patos/flags.md)

    Turn keyword options into a clean CLI argv tuple.

-   ### [strflag](patos/strflag.md)

    An enum `Flag` whose members carry a literal string, OR-combinable and iterable.

</div>
