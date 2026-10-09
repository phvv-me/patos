"""The hash tables kernels read, and the hash they share with the host.

A `StaticMap` is cuCollections' `static_map` from 64-bit keys to 64-bit payloads, built on the
device by cuco's insert and read by cuco's `find`; a `PairTable` is the same map read only, each
probe one 16-byte load. A `Bitmap` holds one flag per bit, and a `Filter` is a bitmap of hashed
values whose miss is certain. Each is a record whose device members read like the Python they
mirror: `table.get(key)`, `map.find(key)`, `bit in bitmap`, `value in filter`.
"""

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from ..hashing import EMPTY_KEY, GOLDEN_GAMMA, MIX_ONE, MIX_TWO, splitmix, table_capacity
from ..typed import Cxx, Struct, Vector, cuco, device, i64, items, kernel, u32, u64
from .memory import address

# Bits in a filter. Sixty four thousand against a few hundred members keeps a stray hit under
# one in a hundred, and the eight kilobytes sit in every SM's first-level cache.
_FILTER_BITS = 1 << 16


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_MAP = Cxx(
    cuco,
    r"""
#include <cuco/static_map_ref.cuh>
#include <cuda/std/functional>

using Key = cuda::std::uint64_t;
using Payload = cuda::std::int64_t;
using Slot = cuco::pair<Key, Payload>;
using Extent = cuco::extent<cuda::std::uint32_t>;
using Storage = cuco::bucket_storage_ref<Slot, 1, Extent>;

// splitmix64, as `device_splitmix` and the host's `splitmix` hash.
struct Splitmix {
  __host__ __device__ constexpr Key operator()(Key key) const noexcept
  {
    Key value = key + 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
  }
};
using Probing = cuco::linear_probing<1, Splitmix>;

// `1 << shift` slots, a power of two the compiler sees, so cuco's modulo compiles to a mask.
__device__ Extent extent(cuda::std::uint32_t shift) { return Extent{1u << shift}; }

template <class... Operators>
__device__ auto map(cuda::std::uint64_t slots, cuda::std::uint32_t shift)
{
  using Map = cuco::static_map_ref<Key, Payload, cuda::thread_scope_device,
                                   cuda::std::equal_to<Key>, Probing, Storage, Operators...>;
  return Map{cuco::empty_key<Key>{~Key{0}}, cuco::empty_value<Payload>{-1}, {}, {},
             cuco::thread_scope_device, Storage{extent(shift), reinterpret_cast<Slot*>(slots)}};
}
""",
    name="cuco_map",
)


@_MAP("""
  auto held = map<cuco::op::insert_tag>(slots, shift);
  static_cast<void>(held.insert(Slot{key, Payload(payload)}));
""")
def _inserted(slots: u64, shift: u32, key: u64, payload: u64) -> None:
    """Insert `key` with `payload` into the map at `slots`, unless it holds the key already."""
    raise NotImplementedError


@_MAP("""
  auto const held = map<cuco::op::find_tag>(slots, shift);
  auto const found = held.find(key);
  return found == held.end() ? Payload{-1} : found->second;
""")
def _found(slots: u64, shift: u32, key: u64) -> i64:
    """What cuco's `find` answers for `key` in the map at `slots`, -1 for no payload."""
    raise NotImplementedError


@_MAP("""
  auto probe = Probing{}.make_iterator<1>(key, extent(shift));
  auto const* pairs = reinterpret_cast<ulonglong2 const*>(slots);
  while (true) {
    // `memory.load_pair`'s PTX: `__ldg` of a `ulonglong2` compiles to two 8-byte loads on sm_121.
    ulonglong2 slot;
    asm("ld.global.nc.v2.u64 {%0, %1}, [%2];" : "=l"(slot.x), "=l"(slot.y) : "l"(pairs + *probe));
    // One test ends the probe: an empty slot's payload is the empty value, -1. The payload is
    // moved out of the load's aligned register quad, which answers held at once would pin
    // (cutok's short merge: 130 registers, not 96); a select instead costs 3% on the GH200.
    if (slot.x == key || slot.x == ~Key{0}) {
      Payload answer;
      asm("mov.b64 %0, %1;" : "=l"(answer) : "l"(slot.y));
      return answer;
    }
    ++probe;
  }
""")
def _probed(slots: u64, shift: u32, key: u64) -> i64:
    """Answer as `_found` does, loading each slot of cuco's probe sequence once, read-only."""
    raise NotImplementedError


@device
def device_splitmix(key: u64) -> u64:
    """`splitmix` over a `u64` key on the device, bit for bit the host body."""
    value = (key + GOLDEN_GAMMA) & EMPTY_KEY
    value ^= value >> 30
    value = (value * MIX_ONE) & EMPTY_KEY
    value ^= value >> 27
    value = (value * MIX_TWO) & EMPTY_KEY
    return value ^ (value >> 31)


class StaticMap(Struct):
    """cuco's `static_map` from 64-bit keys to 64-bit payloads, built on the device by cuco's
    insert and read as `map.find(key)` by cuco's own `find`.

    slots: the (key, payload) pairs, 16 bytes each and all ones where empty, a key placed by
        splitmix64 among `1 << shift` slots and probed linearly from there.
    shift: the capacity's base-two logarithm.
    """

    slots: Vector[u64]
    shift: u32

    @classmethod
    def build(cls, entries: Mapping[int, int]) -> StaticMap:
        """The map of `entries`, payloads by key, inserted on the device at most half full.

        A key is never all ones; a payload is an unsigned 64-bit value, answered as an `i64`.
        Entries go in their mapping's order, in rounds of doubling size, so an earlier one sits
        nearer its home slot: listed most-read first, the first thousand of a merge table's
        pairs take 1.01 probes, against 1.48 inserted all at once (gpt2).
        """
        count = len(entries)
        shift = table_capacity(count).bit_length() - 1
        built = cls(cp.full(2 << shift, EMPTY_KEY, cp.uint64), shift)
        keys, payloads = (
            cp.asarray(np.fromiter(column, np.uint64, count))
            for column in (entries.keys(), entries.values())
        )
        start, size = 0, 64
        while start < count:
            batch = slice(start, start + size)
            built.insert[len(keys[batch])](keys[batch], payloads[batch])
            start, size = start + size, 2 * size
        return built

    @device
    def find(self, key: u64) -> i64:
        """The payload `key` maps to, or -1 when the map holds none."""
        return _found(address(self.slots, 0), self.shift, key)

    @kernel(threads=256)
    def insert(self, keys: Vector[u64], payloads: Vector[u64]) -> None:
        """Insert each key with its payload, a key the map already holds keeping its own."""
        for item in items(keys.size):
            _inserted(address(self.slots, 0), self.shift, keys[item], payloads[item])


class PairTable(Struct):
    """A `StaticMap` no kernel writes while it reads, read as `table.get(key)`.

    It walks cuco's probe sequence with each slot one read-only 16-byte load and one test, where
    `find` loads the key and then the payload: 1.6 times slower on hits the cache holds.

    slots, shift: a `StaticMap`'s; the slots start on a 16-byte boundary, as every CuPy
        allocation does.
    """

    slots: Vector[u64]
    shift: u32

    @classmethod
    def build(cls, entries: Mapping[int, int]) -> PairTable:
        """The table of `entries`, payloads by key, built as `StaticMap.build` builds a map."""
        return cls.of(StaticMap.build(entries))

    @device
    def get(self, key: u64) -> i64:
        """The payload `key` maps to, or -1 when the table holds none."""
        return _probed(address(self.slots, 0), self.shift, key)


class Bitmap(Struct):
    """A flag per bit packed sixty-four to a word, asked `bit in bitmap`; no bit lies past it."""

    words: Vector[u64]

    @device
    def __contains__(self, bit: u64) -> bool:
        return (self.words[bit >> 6] >> (bit & 63)) & 1 != 0

    @classmethod
    def pack(cls, flags: np.ndarray) -> Bitmap:
        """The bitmap whose bit `i` is `flags[i]`, packed into `uint64` words and uploaded.

        flags: a `bool` array with shape `[n]`, `n` a multiple of 64.
        """
        return cls(np.packbits(flags, bitorder="little").view(u64))


class Filter(Struct):
    """A one-hash bitmap holding every value it was built from, and a few it never was.

    A miss is certain and a hit is a hint, which is the whole use: a probe into a pair table is
    skipped only when the filter says the key cannot be there.
    """

    bits: Bitmap

    @device
    def __contains__(self, value: u64) -> bool:
        return (device_splitmix(value) & (_FILTER_BITS - 1)) in self.bits

    @classmethod
    def build(cls, values: Iterable[int]) -> Filter:
        """The filter of `values`, uploaded."""
        flags = np.zeros(_FILTER_BITS, dtype=np.bool_)
        flags[[splitmix(value) & (_FILTER_BITS - 1) for value in values]] = True
        return cls(Bitmap.pack(flags))
