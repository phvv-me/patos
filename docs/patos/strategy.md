# strategy

A named family of interchangeable implementations, selected at runtime.

## When to reach for it

Reach for `strategy` when one job has several swappable implementations and the choice is made at
runtime, by config, by environment, or by what happens to be installed. Compression backends,
serializers, schedulers, accelerated kernels with a pure-Python fallback. You register each
implementation under a name and ask for one by name, or ask for the first one that is available.

## The anti-pattern it replaces

It replaces the import-time `try: import fast except ImportError: import slow` dance and the
module-level global that holds the chosen backend. It also replaces passing bare callables around
with no shared contract. `strategy` gives the family a typed home, a name per implementation, and
a first-available mode so optional backends degrade gracefully.

## Usage

```python
from patos import Strategy


class Compressor(Strategy["Compressor"]):
    def compress(self, data: bytes) -> bytes: ...


@Compressor.register("zstd")
class Zstd(Compressor):
    def compress(self, data: bytes) -> bytes:
        return zstd_compress(data)


@Compressor.register("gzip")
class Gzip(Compressor):
    def compress(self, data: bytes) -> bytes:
        return gzip_compress(data)


Compressor.get("zstd")        # the Zstd implementation, by name
Compressor.first_available()  # the first registered impl that imports and loads
```

## Public API

- `Strategy[T]`. Generic base. Parametrize it with the family's own type for typed lookups.
- `Strategy.register(name)`. Class decorator that enrolls an implementation under a name.
- `Strategy.get(name)`. Fetch one implementation by name.
- `Strategy.first_available()`. Return the first registered implementation that loads, for optional-backend selection.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/strategy.py"
```
