"""Device building blocks kernels share, called through their module (`warp.sum(x)`): the
warp-wide and block-wide reductions numba-cuda lacks (`warp`, and `block.over(threads)` for a
block), `scalar` operations, binary `search`, `bits`, `bytewise` and `memory` operations on the
words of a text, and the open-addressed tables the host builds and kernels read as records
(`PairTable`, `Bitmap`, `Filter`)."""

from ..hashing import EMPTY_KEY, build_pair_slots, find_pair, pair_key, splitmix, table_capacity
from . import bits, block, bytewise, memory, scalar, search, warp
from .hash import Bitmap, Filter, PairTable, device_splitmix

__all__ = [
    "EMPTY_KEY", "Bitmap", "Filter", "PairTable", "build_pair_slots", "device_splitmix",
    "block", "find_pair", "pair_key", "scalar", "search", "splitmix", "table_capacity", "warp",
    "bits", "bytewise", "memory",
]  # fmt: skip
