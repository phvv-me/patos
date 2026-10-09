"""The `u32` and `u64` byte operations of `patos.cuda.primitives.bytewise`, against the host."""

from collections.abc import Callable
from functools import cache

import cupy as cp
import numpy as np
import pytest

from patos.cuda.primitives import bytewise
from patos.cuda.typed import Kernel, Vector, items, kernel, number, u8, unsigned

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

_COUNT = 4096
# Bytes that make matches, ranges and UTF-8 classes frequent in a random word.
_ALPHABET = np.array(
    [0, 1, 0x20, 0x2F, 0x30, 0x39, 0x3A, 0x41, 0x5A, 0x7F, 0x80, 0xBF, 0xC0, 0xC3, 0xE2, 0xFF],
    dtype=np.uint8,
)


@cache
def applying(name: str) -> Kernel:
    """A kernel giving `held[i]` the bytewise function `name` of `words[i]`."""
    function = getattr(bytewise, name)

    @kernel
    def apply(words: Vector[unsigned], held: Vector[number]) -> None:
        for item in items(words.size):
            held[item] = function(words[item])

    return apply


@cache
def applying_with(name: str) -> Kernel:
    """A kernel giving `held[i]` the bytewise function `name` of `words[i]` and a byte."""
    function = getattr(bytewise, name)

    @kernel
    def apply(words: Vector[unsigned], held: Vector[number], byte: u8) -> None:
        for item in items(words.size):
            held[item] = function(words[item], byte)

    return apply


@cache
def applying_between(name: str) -> Kernel:
    """A kernel giving `held[i]` the bytewise function `name` of `words[i]` and two bounds."""
    function = getattr(bytewise, name)

    @kernel
    def apply(words: Vector[unsigned], held: Vector[number], bounds: Vector[u8]) -> None:
        for item in items(words.size):
            held[item] = function(words[item], bounds[0], bounds[1])

    return apply


def _words_of(kind: type[np.unsignedinteger], *, plain: bool = False) -> np.ndarray:
    """Words of mostly alphabet bytes, all below 0x80 when `plain`, led by 0 and the largest."""
    rng, size = np.random.default_rng(7), np.dtype(kind).itemsize
    pool = _ALPHABET[_ALPHABET < 0x80] if plain else _ALPHABET
    noise = rng.integers(0, 0x80 if plain else 256, size=(_COUNT, size), dtype=np.uint8)
    chosen = np.where(rng.random((_COUNT, size)) < 0.7, rng.choice(pool, (_COUNT, size)), noise)
    words = chosen.astype(np.uint8).view(kind).ravel()
    words[:2] = [0, np.iinfo(kind).max & (0x7F7F7F7F7F7F7F7F if plain else -1)]
    return words


def _bytes_of(words: np.ndarray) -> np.ndarray:
    """The bytes of each word in memory order."""
    return words.view(np.uint8).reshape(len(words), -1)


def _flagged(flags: np.ndarray) -> list[int]:
    """The mask with 0x80 in every byte its row of `flags` holds true."""
    tops = np.uint64(8) * np.arange(flags.shape[1], dtype=np.uint64) + np.uint64(7)
    return (flags.astype(np.uint64) << tops).sum(axis=1, dtype=np.uint64).tolist()


def _launched(launch: Kernel, words: np.ndarray, *extra, dtype=np.uint64) -> list[int]:
    """What the kernel `launch` gave each of `words`, `extra` being the arguments after them."""
    held = cp.zeros(len(words), dtype)
    launch[len(words)](cp.asarray(words), held, *extra)
    return held.get().tolist()


@pytest.mark.parametrize("kind", [np.uint32, np.uint64])
@pytest.mark.parametrize(
    ("name", "holds"),
    [
        ("zeros", lambda byte: byte == 0),
        ("ascii", lambda byte: byte < 0x80),
        ("continuations", lambda byte: (byte & 0xC0) == 0x80),
        ("openings", lambda byte: (byte & 0xC0) != 0x80),
    ],
)
def test_a_flagging_function_sets_the_top_bit_of_the_bytes_the_host_flags(
    kind: type[np.unsignedinteger], name: str, holds: Callable
) -> None:
    """No bit is set past the word's width, so a `u32` mask is no `u64` one."""
    words = _words_of(kind)
    assert _launched(applying(name), words) == _flagged(holds(_bytes_of(words)))


@pytest.mark.parametrize("kind", [np.uint32, np.uint64])
@pytest.mark.parametrize("needle", [0x00, 0x20, 0xC3, 0xFF])
def test_equal_flags_the_bytes_that_are_the_needle(
    kind: type[np.unsignedinteger], needle: int
) -> None:
    words = _words_of(kind)
    assert _launched(applying_with("equal"), words, needle) == _flagged(_bytes_of(words) == needle)


@pytest.mark.parametrize("kind", [np.uint32, np.uint64])
@pytest.mark.parametrize("bounds", [(0x30, 0x39), (0x41, 0x5A), (0x00, 0x7F), (0x7F, 0x7F)])
def test_within_flags_the_ascii_bytes_in_the_range(
    kind: type[np.unsignedinteger], bounds: tuple[int, int]
) -> None:
    plain, given = _words_of(kind, plain=True), cp.asarray(np.array(bounds, dtype=np.uint8))
    inside = (_bytes_of(plain) >= bounds[0]) & (_bytes_of(plain) <= bounds[1])
    assert _launched(applying_between("within"), plain, given) == _flagged(inside)


@pytest.mark.parametrize(
    ("kind", "index"),
    [(np.uint32, 0), (np.uint32, 3), (np.uint64, 3), (np.uint64, 4), (np.uint64, 7)],
)
def test_byte_picks_the_byte_at_the_index(kind: type[np.unsignedinteger], index: int) -> None:
    words = _words_of(kind)
    assert _launched(applying_with("byte"), words, index) == _bytes_of(words)[:, index].tolist()


@pytest.mark.parametrize("kind", [np.uint32, np.uint64])
def test_a_mask_counts_its_flagged_bytes_and_gives_the_first_and_last_or_minus_one(
    kind: type[np.unsignedinteger],
) -> None:
    """Rows flag none, a few, half or every byte."""
    odds = np.random.default_rng(7).choice([0, 0.15, 0.5, 1], (_COUNT, 1))
    flags = np.random.default_rng(8).random((_COUNT, np.dtype(kind).itemsize)) < odds
    masks, where = np.array(_flagged(flags), dtype=kind), [np.flatnonzero(row) for row in flags]
    held = {
        name: _launched(applying(name), masks, dtype=np.int64)
        for name in ("count", "first", "last")
    }
    assert held["count"] == [len(row) for row in where]
    assert held["first"] == [row[0] if len(row) else -1 for row in where]
    assert held["last"] == [row[-1] if len(row) else -1 for row in where]


def test_spreading_a_byte_fills_the_eight_bytes_of_a_word() -> None:
    bytes_in = np.arange(256, dtype=np.uint8)
    assert _launched(applying("spread"), bytes_in) == [
        byte * 0x0101010101010101 for byte in range(256)
    ]
