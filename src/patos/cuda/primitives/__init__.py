"""Device building blocks kernels share, called through their module (`warp.sum(x)`): the
warp-wide and block-wide reductions numba-cuda lacks (`warp`, and `block.over(threads)` for a
block), `scalar` operations, binary `search`, `bits`, `bytewise` and `memory` operations on the
words of a text, and the hash tables kernels read as records (`StaticMap` and `PairTable` on
cuCollections, `Bitmap`, `Filter`)."""

from ..hashing import EMPTY_KEY, pair_key, splitmix, table_capacity
from . import bits, block, bytewise, memory, scalar, search, warp
from .hash import Bitmap, Filter, PairTable, StaticMap, device_splitmix

__all__ = [
    "EMPTY_KEY", "Bitmap", "Filter", "PairTable", "StaticMap", "device_splitmix", "block",
    "pair_key", "scalar", "search", "splitmix", "table_capacity", "warp", "bits", "bytewise",
    "memory",
]  # fmt: skip
