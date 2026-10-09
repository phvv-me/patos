"""The bit operations and aligned loads of `patos.cuda.primitives` and the stubs under them."""

from typing import NamedTuple

import cupy as cp
import numpy as np
import pytest

from patos.cuda.primitives import bits, memory
from patos.cuda.typed import AnnotationError, Kernel, items, kernel, ptx, u8, u32, u64

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

_COUNT = 4096


@ptx("add.u32 $result0, $value, $value;\nmov.u64 $result1, $wide;", pure=True)
def doubled_and_wide(value: u32, wide: u64) -> tuple[u32, u64]:
    """`value` doubled, and `wide` as it is: a tuple of two types."""
    raise NotImplementedError


@ptx("st.global.u32 [$at], $value;")
def store32(at: u64, value: u32) -> None:
    """Store `value` at the 4-aligned address `at`."""
    raise NotImplementedError


@kernel
def effects(narrow: u32[int], wide: u64[int], held: u64[int, int], stored: u32[int]) -> None:
    for item in items(wide.size):
        doubled, again = doubled_and_wide(narrow[item], wide[item])
        held[item, 0], held[item, 1] = doubled, again
        store32(memory.address(stored, item), doubled)


@kernel
def mixing(table: u32[int, int]) -> None:
    """Columns 0 to 2 are the operands, and 3 to 6 get `permute`, `funnel`, `dot4`, `absdiff4`."""
    for item in items(table.shape[0]):
        first, second, third = table[item, 0], table[item, 1], table[item, 2]
        table[item, 3] = bits.permute(first, second, third)
        table[item, 4] = bits.funnel(first, second, third)
        table[item, 5] = bits.dot4(first, second, third)
        table[item, 6] = bits.absdiff4(first, second)


@kernel
def loading(wide: u64[int], narrow: u32[int], held: u64[int, int]) -> None:
    """Row `g` holds the word and the quad at element `4g` of `narrow`, the pair at `2g`."""
    for group in items(held.shape[0]):
        first, second, third, fourth = memory.load_quad(memory.address(narrow, 4 * group))
        low, high = memory.load_pair(memory.address(wide, 2 * group))
        held[group, 0] = memory.load32(memory.address(narrow, 4 * group))
        held[group, 1] = memory.load64(memory.address(wide, 2 * group))
        held[group, 2], held[group, 3], held[group, 4], held[group, 5] = (
            first,
            second,
            third,
            fourth,
        )
        held[group, 6], held[group, 7], held[group, 8] = low, high, memory.address(narrow, group)


@kernel
def windowing(chars: u8[int], starts: u64[int], held: u64[int, int]) -> None:
    for item in items(starts.size):
        held[item, 0], held[item, 1] = memory.window(chars, starts[item])


def _ptx_of(launched: Kernel) -> str:
    """The PTX `launched` compiled to, once it has been launched."""
    return launched.dispatcher.inspect_asm(next(iter(launched.dispatcher.overloads)))


def _bytes_of(words: np.ndarray) -> np.ndarray:
    """The bytes of each word in memory order."""
    return np.ascontiguousarray(words).view(np.uint8).reshape(len(words), -1)


def _permuted(table: np.ndarray) -> np.ndarray:
    """PTX `prmt.b32` of the first three columns of `table`, a uint32 array with shape `[n, 7]`.

    Nibble `i` of the selector fills byte `i` with a byte of `second:first`, or with that byte's
    sign bit when the nibble's top bit is set.
    """
    source = np.concatenate([_bytes_of(table[:, 0]), _bytes_of(table[:, 1])], axis=1)
    selector = table[:, 2:3]
    picks = (selector >> (4 * np.arange(4))) & 0xF
    chosen = np.take_along_axis(source, (picks & 7).astype(np.intp), axis=1).astype(np.int64)
    filled = np.where(picks & 8, np.where(chosen & 0x80, 0xFF, 0), chosen)
    return filled.astype(np.uint8).view(np.uint32).ravel()


class Mixed(NamedTuple):
    """What `mixing` made of random operands: a uint32 table with shape `[n, 7]`."""

    table: np.ndarray


@pytest.fixture
def mixed() -> Mixed:
    operands = np.random.default_rng(7).integers(0, 2**32, (_COUNT, 3))
    table = cp.zeros((_COUNT, 7), np.uint32)
    table[:, :3] = cp.asarray(operands, np.uint32)
    mixing[_COUNT](table)
    return Mixed(table.get())


def test_permute_and_funnel_give_the_bytes_and_bits_the_host_picks(mixed: Mixed) -> None:
    table = mixed.table
    both = (table[:, 1].astype(np.uint64) << 32) | table[:, 0]
    assert table[:, 3].tolist() == _permuted(table).tolist()
    assert table[:, 4].tolist() == ((both >> (table[:, 2] & 31)) & 0xFFFFFFFF).tolist()


def test_dot4_and_absdiff4_combine_the_bytes_of_two_words(mixed: Mixed) -> None:
    table = mixed.table
    left, right = (_bytes_of(table[:, column]).astype(np.int64) for column in (0, 1))
    wanted = np.abs(left - right).astype(np.uint8).view(np.uint32).ravel()
    assert table[:, 5].tolist() == (((left * right).sum(axis=1) + table[:, 2]) % 2**32).tolist()
    assert table[:, 6].tolist() == wanted.tolist()


@pytest.mark.parametrize(
    "instruction", ["prmt.b32", "shf.r.wrap.b32", "dp4a.u32.u32", "vabsdiff4.u32.u32.u32"]
)
def test_a_bit_operation_is_its_one_ptx_instruction(mixed: Mixed, instruction: str) -> None:
    assert _ptx_of(mixing).count(instruction) == 1


def test_a_stub_returns_a_tuple_of_registers_or_nothing() -> None:
    """Two registers of two types come back as a tuple, and a store stays an effect."""
    rng = np.random.default_rng(7)
    narrow = rng.integers(0, 2**32, 256, dtype=np.uint32)
    wide = rng.integers(0, 2**64, 256, dtype=np.uint64)
    held, stored = cp.zeros((256, 2), np.uint64), cp.zeros(256, np.uint32)
    effects[256](cp.asarray(narrow), cp.asarray(wide), held, stored)

    doubled = (narrow * 2).astype(np.uint32)
    assert held.get().tolist() == np.stack([doubled, wide], 1).tolist()
    assert stored.get().tolist() == doubled.tolist()
    assert "st.global.u32" in _ptx_of(effects)


def test_a_tuple_of_bytes_is_refused_where_the_stub_is_defined() -> None:
    def bytes_back(at: u64) -> tuple[u32, u8]:
        raise NotImplementedError

    with pytest.raises(AnnotationError, match="16, 32 or 64 bits"):
        ptx("mov.b32 $result, $at;")(bytes_back)


def test_an_aligned_load_reads_the_words_the_host_holds_in_one_instruction() -> None:
    """The word, the quad and the pair at every 16-byte group, and the address of an element."""
    wide = cp.asarray(np.random.default_rng(7).integers(0, 2**64, 1024, dtype=np.uint64))
    held = cp.zeros((512, 9), np.uint64)
    loading[512](wide, wide.view(cp.uint32), held)

    host, groups = wide.get(), np.arange(512)
    narrow = host.view(np.uint32)
    columns = [
        narrow[4 * groups], host[2 * groups], *(narrow[4 * groups + step] for step in range(4)),
        host[2 * groups], host[2 * groups + 1], wide.data.ptr + 4 * groups,
    ]  # fmt: skip
    assert held.get().tolist() == np.array(columns, dtype=np.uint64).T.tolist()
    sizes = ["v4.u32", "v2.u64", "u32 ", "u64 "]
    assert [_ptx_of(loading).count(f"ld.global.nc.{size}") for size in sizes] == [1] * 4


@pytest.mark.parametrize("offset", [0, 1, 3, 5])
@pytest.mark.parametrize("size", [1, 2, 7, 8, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 33, 257])
def test_a_window_is_the_sixteen_bytes_at_any_offset_and_zero_past_the_end(
    *, size: int, offset: int
) -> None:
    """Every start in arrays whose length is no multiple of 4, 8 or 16, off every alignment."""
    data = np.random.default_rng(size).integers(0, 256, offset + size + 8, dtype=np.uint8)
    chars, starts = cp.asarray(data)[offset : offset + size], cp.arange(size, dtype=cp.uint64)
    held = cp.zeros((size, 2), np.uint64)
    windowing[size](chars, starts, held)

    padded = np.concatenate([chars.get(), np.zeros(16, np.uint8)])
    around = np.arange(size)[:, None] + np.arange(16)[None, :]
    assert held.get().tolist() == padded[around].view(np.uint64).tolist()
