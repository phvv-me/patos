<div class="hero" markdown>

![patos](assets/banner.png){ .hero-banner }

<p class="tagline">A flock of typed, zero-dependency Python patterns. Pip one in, or copy one in and own the file.</p>

[Get started](#two-ways-to-use-a-pato){ .md-button .md-button--primary }
[Browse the flock](#the-flock){ .md-button }

</div>

Each pattern is a single self-contained `pato` (a duck). Adopt one, and you get a tiny,
fully typed, standard-library-only module that does one thing well. The flock is `patos`. A
single duck is a `pato`.

## Two ways to use a pato

<div class="ways" markdown>

<div class="way" markdown>
<span class="pill">pip it in</span>

Install the whole flock and import the patterns you want.

```sh
pip install patos
```

```python
from patos import Registry, Singleton
```
</div>

<div class="way" markdown>
<span class="pill">copy it in</span>

Open a duck's page, copy its source from the **Source** section, and paste it into your own tree.
No runtime dependency, no version pin, no tool, just one module you own.

```python
# paste src/patos/singleton.py
# the file is yours now
```
</div>

</div>

## Why patos

- **Typed.** Every pato ships full type hints and `py.typed`, so your checker sees through it.
- **Zero dependencies.** Each duck is standard library only. Copying one in adds nothing to your lockfile.
- **You own the code.** The copy-in path gives you a small, readable module instead of a black box.
- **One idea per duck.** No god-objects, no framework. Reach for the one pattern you need.

## The flock

### Creational

<div class="grid cards" markdown>

-   ### [singleton](patos/singleton.md)

    One instance per class, with `__init__` running exactly once. Metaclass based, never a
    cached `__new__`.

-   ### [flyweight](patos/flyweight.md)

    A metaclass that interns instances by their constructor args. Build once, share forever, stay
    immutable.

</div>

### Dispatch & selection

<div class="grid cards" markdown>

-   ### [registry](patos/registry.md)

    Self-registering class hierarchies. Subclasses enroll through `__init_subclass__`, and
    `dispatch` tries each one until it builds.

-   ### [strategy](patos/strategy.md)

    A named family of interchangeable implementations, picked at runtime or by
    `first_available()`.

-   ### [dispatch](patos/dispatch.md)

    Dispatch on a value the way `singledispatch` dispatches on a type.

</div>

### Command-line

<div class="grid cards" markdown>

-   ### [strflag](patos/strflag.md)

    An enum `Flag` whose members carry a literal string, OR-combinable and iterable.

</div>
