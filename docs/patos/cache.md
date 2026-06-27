# cache

A load-once cache of a derived value, keyed by exactly the things the value depends on.

## When to reach for it

Reach for `DerivedCache` when an expensive artifact (a factorization, a covariance, a
deterministic evaluation) depends not on a whole config but on a few of its fields. Key the
cache by that subset, and two callers sharing the subset share one build.

## The anti-pattern it replaces

It replaces the bespoke `if key not in self.store: self.store[key] = build()` memo each
load-once state re-invents, and the over-broad cache keyed by the full config that recomputes
when only an irrelevant field changed. Declare the key, and `get` computes once then reuses.

## Usage

```python
from patos import DerivedCache


cache: DerivedCache[tuple[str, int], str] = DerivedCache()


def factorize(role: str, rank: int) -> str:
    ...  # the expensive build


cache.get(("q", 4), lambda: factorize("q", 4))   # builds
cache.get(("q", 4), lambda: factorize("q", 4))   # cache hit, build skipped
len(cache)                                        # 1
("q", 4) in cache                                 # True
```

## Public API

- `DerivedCache()`. An empty cache, generic over the key and the derived value.
- `DerivedCache.get(key, build)`. The cached value for `key`, computing it through `build` once on the first miss.

## Source

Copy this into your project and own it. No dependency, no tool, just one module you can read and change.

```python
--8<-- "src/patos/cache.py"
```
