"""The bit operations, the SIMD operations on packed lanes and the aligned loads of
`patos.cuda.primitives`, and the stubs under them."""

import re
from collections import Counter
from collections.abc import Callable, Sequence
from functools import cache
from typing import TYPE_CHECKING, NamedTuple

import cupy as cp
import numpy as np
import pytest

from patos.cuda.primitives import bits, memory, warp
from patos.cuda.typed import (
    AnnotationError,
    Kernel,
    Matrix,
    Vector,
    cuda,
    device,
    dispatched,
    i8x4,
    i16x2,
    i32,
    intrinsics,
    items,
    kernel,
    ptx,
    u8,
    u8x4,
    u16x2,
    u32,
    u64,
    unsigned,
)

if TYPE_CHECKING:
    from patos.cuda.scalars import Packed

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

_COUNT = 4096
# Words whose lanes sit at the edges of every lane type.
_EDGES = [0, 0xFFFFFFFF, 0x80808080, 0x7F7F7F7F, 0x80008000, 0x7FFF7FFF, 0x00FF00FF, 0x01017F80]
# What a lane operation reads a word as: lanes, or a 32-bit integer.
type Operand = type[Packed] | type[u32] | type[i32]
# Each operation over what it reads its first two words as, and the SASS instructions it adds to
# a kernel from sm_86 to sm_90 and on sm_121, where the video instructions are emulated.
_SASS: dict[tuple[str, tuple[Operand, Operand]], tuple[int, int]] = {
    ("absdiff", (u8x4, u8x4)): (1, 3), ("absdiff", (i8x4, i8x4)): (3, 5),
    ("absdiff", (u16x2, u16x2)): (7, 11), ("absdiff", (i16x2, i16x2)): (7, 11),
    ("absdiff", (u32, u32)): (1, 3), ("absdiff", (i32, i32)): (1, 3),
    ("sad", (u8x4, u8x4)): (1, 6), ("sad", (i8x4, i8x4)): (3, 8),
    ("sad", (u32, u32)): (1, 3), ("sad", (i32, i32)): (1, 3),
    ("dot", (u8x4, u8x4)): (1, 1), ("dot", (i8x4, i8x4)): (1, 1),
    ("dot", (u16x2, u8x4)): (1, 1), ("dot", (i16x2, i8x4)): (1, 1),
    ("min", (u8x4, u8x4)): (6, 7), ("max", (u8x4, u8x4)): (6, 7),
}  # fmt: skip
_SUMMING = {"sad", "dot"}
# Every opcode patos's templates use, and the effects no template may move, by whether a template
# of it alone is pure.
_OPCODES = {
    "add.s32": True, "add.u32": True, "dp2a.lo.s32.s32": True, "dp2a.lo.u32.u32": True,
    "dp4a.s32.s32": True, "dp4a.u32.u32": True, "mov.s32": True, "mov.u32": True,
    "mov.u64": True, "prmt.b32": True, "sad.s32": True, "sad.u32": True, "shf.r.wrap.b32": True,
    "vabsdiff4.u32.u32.u32": True, "vabsdiff4.u32.u32.u32.add": True,
    "ld.global.nc.u32": False, "ld.global.nc.u64": False, "ld.global.nc.v2.u64": False,
    "ld.global.nc.v4.u32": False, "redux.sync.add.s32": False, "redux.sync.add.u32": False,
    "redux.sync.and.b32": False, "redux.sync.max.s32": False, "redux.sync.max.u32": False,
    "redux.sync.min.s32": False, "redux.sync.min.u32": False, "redux.sync.or.b32": False,
    "redux.sync.xor.b32": False, "shfl.sync.up.b32": False, "st.global.u32": False,
    "atom.global.add.u32": False, "bar.sync": False, "membar.gl": False, "nanosleep.u32": False,
    "vote.sync.ballot.b32": False,
}  # fmt: skip


@ptx("add.u32 $result0, $value, $value;\nmov.u64 $result1, $wide;")
def doubled_and_wide(value: u32, wide: u64) -> tuple[u32, u64]:
    """`value` doubled, and `wide` as it is: a tuple of two types."""
    raise NotImplementedError


@ptx("st.global.u32 [$at], $value;")
def store32(at: u64, value: u32) -> None:
    """Store `value` at the 4-aligned address `at`."""
    raise NotImplementedError


@kernel
def effects(
    narrow: Vector[u32], wide: Vector[u64], held: Matrix[u64], stored: Vector[u32]
) -> None:
    for item in items(wide.size):
        doubled, again = doubled_and_wide(narrow[item], wide[item])
        held[item, 0], held[item, 1] = doubled, again
        store32(memory.address(stored, item), doubled)


@kernel
def mixing(table: Matrix[u32]) -> None:
    """Columns 0 to 2 are the operands, and 3 and 4 get `permute` and `funnel`."""
    for item in items(table.shape[0]):
        first, second, third = table[item, 0], table[item, 1], table[item, 2]
        table[item, 3] = bits.permute(first, second, third)
        table[item, 4] = bits.funnel(first, second, third)


@device
def _exclusive(a: u32, b: u32) -> u32:
    """One instruction over the two words a lane operation reads, for the kernel around it."""
    return a ^ b


@device
def _exclusive_summed(a: u32, b: u32, c: u32) -> u32:
    """One instruction over the three words a summing lane operation reads."""
    return a ^ b ^ c


@cache
def applying(operation: Callable, lanes: Sequence[Operand], *, summed: bool) -> Kernel:
    """A kernel giving `out[i]` the `operation` of `words[0, i]` and `words[1, i]`.

    lanes: what the two words are read as.
    summed: whether the operation adds a third word, `words[2, i]`.
    """
    left, right = lanes

    @kernel
    def apply(words: Matrix[unsigned], out: Vector[unsigned]) -> None:
        for item in items(out.size):
            out[item] = operation(left(words[0, item]), right(words[1, item]))

    @kernel
    def accumulate(words: Matrix[unsigned], out: Vector[unsigned]) -> None:
        for item in items(out.size):
            out[item] = operation(left(words[0, item]), right(words[1, item]), words[2, item])

    return accumulate if summed else apply


@kernel
def loading(wide: Vector[u64], narrow: Vector[u32], held: Matrix[u64]) -> None:
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
def joining(halves: Matrix[u8], words: Matrix[u32], held: Matrix[u64]) -> None:
    for item in items(held.shape[0]):
        held[item, 0] = bits.join(halves[item, 0], halves[item, 1])
        held[item, 1] = bits.join(words[item, 0], words[item, 1])


@kernel
def bit_reading(
    narrow: Vector[u32], wide: Vector[u64], indices: Vector[u32], held: Matrix[u32]
) -> None:
    for item in items(narrow.size):
        held[item, 0] = bits.bit(narrow[item], indices[item] & 31)
        held[item, 1] = bits.bit(wide[item], indices[item])


@kernel
def comparing(left: Vector[u8], right: Vector[u8], ranges: Matrix[u64], held: Vector[u8]) -> None:
    for item in items(held.size):
        held[item] = memory.equal(left, ranges[item, 0], right, ranges[item, 1], ranges[item, 2])


@kernel
def copying[T: (u8, i32)](source: Vector[T], target: Vector[T], ranges: Matrix[u64]) -> None:
    for item in items(ranges.shape[0]):
        memory.copy(source, ranges[item, 0], target, ranges[item, 1], ranges[item, 2])


@kernel
def packing(chars: Vector[u8], ranges: Matrix[u64], held: Matrix[u64]) -> None:
    for item in items(held.shape[0]):
        held[item, 0], held[item, 1] = memory.pack(chars, ranges[item, 0], ranges[item, 1])


@kernel
def windowing(chars: Vector[u8], starts: Vector[u64], held: Matrix[u64]) -> None:
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


def _sass(launched: Kernel) -> Counter[str]:
    """How many times each SASS instruction appears in what `launched` compiled to."""
    sass = launched.dispatcher.inspect_sass(next(iter(launched.dispatcher.overloads)))
    found = re.finditer(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][\w.]*)", sass)
    return Counter(match[1] for match in found if match[1] != "NOP")


def _computed(name: str, lanes: Sequence[Operand], words: np.ndarray) -> np.ndarray:
    """What the host computes for lane operation `name` over the three rows of uint32 `words`.

    Each word reads as a row of int64 lanes, a 32-bit integer as a row of one.
    """
    first, second = (
        row.view(getattr(kind, "element", kind)).reshape(len(row), -1).astype(np.int64)
        for row, kind in zip(words, lanes, strict=False)
    )
    match name:
        case "sad":
            return (words[2] + np.abs(first - second).sum(axis=1)) % 2**32
        case "dot":
            return (words[2] + (first * second[:, : first.shape[1]]).sum(axis=1)) % 2**32
        case "absdiff":
            values = np.abs(first - second)
        case _:
            values = {"min": np.minimum, "max": np.maximum}[name](first, second)
    width = 32 // first.shape[1]
    shifts = np.arange(first.shape[1], dtype=np.uint64) * np.uint64(width)
    return ((values % 2**width).astype(np.uint64) << shifts).sum(axis=1)


def _spelled(value: str | tuple[Operand, Operand]) -> str:
    """A test's id: the operation's name, or how it reads its words."""
    if isinstance(value, str):
        return value
    return "-".join(getattr(kind, "name", None) or kind.__name__ for kind in value)


class Mixed(NamedTuple):
    """What `mixing` made of random operands: a uint32 table with shape `[n, 5]`."""

    table: np.ndarray


@pytest.fixture
def mixed() -> Mixed:
    operands = np.random.default_rng(7).integers(0, 2**32, (_COUNT, 3))
    table = cp.zeros((_COUNT, 5), np.uint32)
    table[:, :3] = cp.asarray(operands, np.uint32)
    mixing[_COUNT](table)
    return Mixed(table.get())


def test_permute_and_funnel_give_the_bytes_and_bits_the_host_picks(mixed: Mixed) -> None:
    table = mixed.table
    both = (table[:, 1].astype(np.uint64) << 32) | table[:, 0]
    assert table[:, 3].tolist() == _permuted(table).tolist()
    assert table[:, 4].tolist() == ((both >> (table[:, 2] & 31)) & 0xFFFFFFFF).tolist()


@pytest.mark.parametrize("instruction", ["prmt.b32", "shf.r.wrap.b32"])
def test_a_bit_operation_is_its_one_ptx_instruction(mixed: Mixed, instruction: str) -> None:
    assert _ptx_of(mixing).count(instruction) == 1


@pytest.mark.parametrize(("name", "lanes"), _SASS, ids=_spelled)
def test_a_lane_operation_gives_what_the_host_computes_lane_by_lane(
    *, name: str, lanes: tuple[Operand, Operand]
) -> None:
    """Random words and every pair of words at the lane types' edges, added to 0xFFFFFFF0."""
    pairs = np.array([*np.meshgrid(_EDGES, _EDGES), np.full((8, 8), 0xFFFFFFF0)], np.uint32)
    random = np.random.default_rng(len(name)).integers(0, 2**32, (3, _COUNT), dtype=np.uint32)
    words = np.concatenate([pairs.reshape(3, -1), random], axis=1)
    out = cp.zeros(words.shape[1], np.uint32)
    applying(getattr(bits, name), lanes, summed=name in _SUMMING)[len(out)](cp.asarray(words), out)

    assert out.get().tolist() == _computed(name, lanes, words).tolist()


@pytest.mark.parametrize(("name", "lanes"), _SASS, ids=_spelled)
def test_a_lane_operation_compiles_to_the_instructions_measured_for_its_lanes(
    *, name: str, lanes: tuple[Operand, Operand]
) -> None:
    """What the operation adds to a kernel over one exclusive or of the same words."""
    capability = cuda.get_current_device().compute_capability
    if capability not in {(8, 6), (8, 9), (9, 0), (12, 1)}:
        pytest.skip(f"no SASS was measured on sm_{capability[0]}{capability[1]}")
    summed = name in _SUMMING
    launched = [
        applying(getattr(bits, name), lanes, summed=summed),
        applying(_exclusive_summed if summed else _exclusive, (i32, i32), summed=summed),
    ]
    words, out = cp.zeros((3, 128), np.uint32), cp.zeros(128, np.uint32)
    for compiled in launched:
        compiled[128](words, out)
    found, base = (_sass(compiled) for compiled in launched)

    assert found.total() - base.total() + 1 == _SASS[name, lanes][capability == (12, 1)]


@pytest.mark.parametrize(("opcode", "pure"), _OPCODES.items())
def test_a_template_is_pure_when_every_instruction_is_register_arithmetic(
    *, opcode: str, pure: bool
) -> None:
    """An opcode is pure alone and after a pure instruction, or an effect in both.

    The table holds every opcode patos's templates use and the effects a template must never move;
    a special register or an opcode patos does not know is an effect too.
    """
    after = f"{{\n.reg .u32 t;\nmov.u32 t, $a;\n@p {opcode} $result, t;\n}}"
    assert intrinsics.is_pure(f"{opcode} $result, $a;") is pure
    assert intrinsics.is_pure(after) is pure
    assert not intrinsics.is_pure("mov.u32 $result, %laneid;")
    assert not intrinsics.is_pure("frobnicate.b32 $result, $a;")


def test_every_template_of_patos_uses_opcodes_the_purity_table_classifies() -> None:
    """A new opcode in a primitive is classified here before it ships."""
    templates = [
        found.template
        for module in (bits, memory, warp)
        for found in vars(module).values()
        if isinstance(getattr(found, "template", None), str)
    ]
    used = {
        match[1]
        for template in templates
        for match in re.finditer(r"(?:^|[;{\n])\s*(?:@!?\w+\s+)?([a-z][\w.]*)\s", template)
    }

    assert len(templates) == 26
    assert used <= set(_OPCODES)


def test_a_dispatched_name_takes_only_implementations_of_its_own_operands() -> None:
    """An implementation of another arity is refused at definition, not at a call."""

    @ptx("add.u32 $result, $a, $b;")
    def added(a: u32, b: u32) -> u32:
        raise NotImplementedError

    def alone(a: u32) -> u32:
        raise NotImplementedError

    with pytest.raises(TypeError, match=r"alone: no implementations take \(u32\)"):
        dispatched(added)(alone)


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


def test_join_puts_the_high_half_above_the_low_one_at_each_width() -> None:
    """Two bytes make a `u16` and two words a `u64`, `low` in the low bits."""
    rng = np.random.default_rng(11)
    edges = [[0, 0], [255, 255], [1, 0], [0, 1]]
    halves = np.vstack([edges, rng.integers(0, 256, (252, 2))]).astype(np.uint64)
    edges = [[0, 2**32 - 1], [2**32 - 1, 0]]
    words = np.vstack([edges, rng.integers(0, 2**32, (254, 2))]).astype(np.uint64)
    held = cp.zeros((256, 2), np.uint64)
    joining[256](cp.asarray(halves, np.uint8), cp.asarray(words, np.uint32), held)

    wanted = np.column_stack([halves[:, 0] | halves[:, 1] << 8, words[:, 0] | words[:, 1] << 32])
    assert held.get().tolist() == wanted.tolist()


def test_bit_is_the_indexed_bit_of_a_word_of_either_width() -> None:
    rng = np.random.default_rng(12)
    narrow = rng.integers(0, 2**32, 512, dtype=np.uint32)
    wide = rng.integers(0, 2**64, 512, dtype=np.uint64)
    indices = rng.integers(0, 64, 512, dtype=np.uint32)
    held = cp.zeros((512, 2), np.uint32)
    bit_reading[512](cp.asarray(narrow), cp.asarray(wide), cp.asarray(indices), held)

    wanted = [
        [int(n) >> (int(i) & 31) & 1, int(w) >> int(i) & 1]
        for n, w, i in zip(narrow, wide, indices, strict=True)
    ]
    assert held.get().tolist() == wanted


def test_equal_compares_two_byte_ranges_up_to_their_first_difference() -> None:
    """Ranges of every length up to 40, equal or differing at one byte, and empty ones."""
    rng = np.random.default_rng(13)
    left = rng.integers(0, 4, 4096, dtype=np.uint8)
    right = left ^ (rng.random(4096) < 0.02)
    starts = rng.integers(0, 4096 - 40, 512)
    others = np.where(np.arange(512) < 256, starts, rng.integers(0, 4096 - 40, 512))
    ranges = np.column_stack([starts, others, rng.integers(0, 41, 512)]).astype(np.uint64)
    held = cp.zeros(512, np.uint8)
    comparing[512](cp.asarray(left), cp.asarray(right), cp.asarray(ranges), held)

    wanted = [bytes(left[a : a + n]) == bytes(right[b : b + n]) for a, b, n in ranges.astype(int)]
    assert held.get().astype(bool).tolist() == wanted


@pytest.mark.parametrize("kind", [np.uint8, np.int32])
def test_copy_moves_each_range_in_one_thread_and_leaves_the_rest(kind: type[np.integer]) -> None:
    """Each item copies up to 40 elements into a slot of its own; what no copy reaches stays."""
    rng = np.random.default_rng(14)
    source = rng.integers(-100, 100, 4096).astype(kind)
    target = np.full(256 * 48, 7, kind)
    ranges = np.column_stack(
        [rng.integers(0, 4096 - 40, 256), np.arange(256) * 48, rng.integers(0, 41, 256)]
    ).astype(np.uint64)
    held = cp.asarray(target)
    copying[256](cp.asarray(source), held, cp.asarray(ranges))

    for start, to, count in ranges.astype(int):
        target[to : to + count] = source[start : start + count]
    assert held.get().tolist() == target.tolist()


def test_pack_gives_up_to_sixteen_bytes_as_two_words_zero_above_them() -> None:
    """Every count from 0 to 16, at any offset; no byte past the range is read into the words."""
    rng = np.random.default_rng(15)
    chars = rng.integers(1, 256, 1024, dtype=np.uint8)
    ranges = np.column_stack([rng.integers(0, 1024 - 16, 340), np.tile(np.arange(17), 20)])
    held = cp.zeros((340, 2), np.uint64)
    packing[340](cp.asarray(chars), cp.asarray(ranges.astype(np.uint64)), held)

    padded = [np.pad(chars[a : a + n], (0, 16 - n)) for a, n in ranges]
    assert held.get().tolist() == np.array(padded).view(np.uint64).tolist()
