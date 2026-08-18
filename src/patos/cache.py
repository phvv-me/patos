import hashlib
from collections.abc import Callable


class DerivedCache[K, V]:
    """A load-once cache of a derived value keyed by whatever it actually depends on.

    The convention a long search reaches for when an expensive artifact (a per-site factorization,
    a pooled covariance, a deterministic per-config evaluation) depends not on a whole config but
    on a few of its fields. Key the cache by exactly that subset and two callers sharing the subset
    share one build, while a resumed or repeated request is attributed to the right cell instead of
    recomputed. :meth:`get` is the whole surface: it returns the cached value for a key, computing
    it through `build` once on the first miss and reusing it forever after.

    Generic over the key `K` (the tuple of the fields the value depends on) and the derived value
    `V`. It owns its store and never evicts, so it is sized by the number of distinct keys seen,
    which is why callers key it by the minimal subset rather than the full config.
    """

    def __init__(self) -> None:
        self.store: dict[K, V] = {}

    def get(self, key: K, build: Callable[[], V]) -> V:
        """The cached value for `key`, computing it through `build` once on the first miss.

        key: identifies the derived value; equal keys share one build.
        build: a zero-arg producer called only when `key` is absent.
        """
        if key not in self.store:
            self.store[key] = build()
        return self.store[key]

    def __len__(self) -> int:
        return len(self.store)

    def __contains__(self, key: K) -> bool:
        return key in self.store

    def __repr__(self) -> str:
        return f"<DerivedCache size={len(self.store)}>"


def content_key(*parts: object) -> str:
    """Hex digest keying a cached artifact by the content that determines it.

    The on-disk counterpart to `DerivedCache`'s in-memory key. Where the cache keys a live
    object by a tuple, a file needs a name, and hashing the defining content into one gives
    it a name that moves whenever that content moves. Any change to a part therefore lands on
    a fresh file instead of silently reusing a stale one.

    parts: the defining content of the artifact, such as an algorithm version, its parameters
        and its source tables. The key is the sha256 of their canonical reprs, so every part
        needs a repr that is stable across processes.
    """
    payload = "\x1f".join(repr(part) for part in parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]
