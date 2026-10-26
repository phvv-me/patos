"""Records built in device code by calling their class, as the host builds them."""

import re
from collections import Counter
from typing import NamedTuple

import compiled_code
import cupy as cp
import numpy as np
import pytest
from numba.core.errors import TypingError
from patos.cuda.typed import (
    Constant,
    Matrix,
    Struct,
    Vector,
    device,
    i32,
    i64,
    items,
    kernel,
    u8,
    u32,
)

_STREAM = cp.cuda.Stream.null

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

# An instruction of a SASS listing, its predicate left out.
_OPCODE = r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][\w.]*)"


class Span(Struct):
    start: u32
    end: u32

    @property
    @device
    def width(self) -> u32:
        return self.end - self.start


class Window(Struct):
    span: Span
    data: Vector[u8]
    step: i32 = 1


class Tiled(Struct):
    tile: Constant[int]
    count: i64


@device
def widened(span: Span, by: i32) -> Span:
    return Span(span.start - by, span.end + by)


@kernel
def windows(starts: Vector[i64], data: Vector[u8], out: Matrix[i64]) -> None:
    for item in items(starts.size):
        window = Window(widened(Span(starts[item], end=starts[item] + 2), 1), data)
        out[item, 0] = window.span.start
        out[item, 1] = window.span.width
        out[item, 2] = window.step
        out[item, 3] = window.data[window.span.end]
        out[item, 4] = Tiled(8, item).tile * Tiled(8, item).count


@kernel
def misbuilt(data: Vector[i32], out: Vector[i32]) -> None:
    for item in items(out.size):
        out[item] = Window(Span(1, 2), data).step


class Pair(NamedTuple):
    low: u32
    high: u32


class Twin(Struct):
    low: u32
    high: u32


@device
def swapped_value(low: u32, high: u32) -> Pair:
    return Pair(high, low)


@device
def swapped_record(low: u32, high: u32) -> Twin:
    return Twin(high, low)


@kernel
def through_value(table: Matrix[u32]) -> None:
    for at in items(table.shape[1]):
        pair = swapped_value(table[0, at], table[1, at])
        table[2, at] = pair.low + pair.high * 3


@kernel
def through_record(table: Matrix[u32]) -> None:
    for at in items(table.shape[1]):
        pair = swapped_record(table[0, at], table[1, at])
        table[2, at] = pair.low + pair.high * 3


def test_device_code_builds_records_by_calling_their_class() -> None:
    """Device code builds a record as the host does.

    Fields arrive by position or name, defaults fill in, scalars convert to their declarations,
    records nest, arrays travel, members answer, and a constant is a literal.
    """
    starts = np.array([3, 10, 20], np.int64)
    data = cp.arange(64, dtype=np.uint8)
    out = cp.zeros((3, 5), np.int64)
    windows[3](cp.asarray(starts), data, out, stream=_STREAM)

    expected = [[start - 1, 4, 1, start + 3, 8 * item] for item, start in enumerate(starts)]
    assert out.get().tolist() == expected


def test_a_field_built_from_a_value_its_declaration_refuses_fails_to_compile() -> None:
    with pytest.raises(TypingError, match=r"Window\.data receives array\(int32"):
        misbuilt[1](cp.zeros(1, np.int32), cp.zeros(1, np.int32), stream=_STREAM)


def test_a_record_built_on_the_device_compiles_as_a_named_value_does() -> None:
    """A record stays in registers, its SASS the named value's instruction for instruction."""
    table = cp.asarray(np.arange(96, dtype=np.uint32).reshape(3, 32))
    held = table.copy()
    through_value[32](table, stream=_STREAM)
    through_record[32](held, stream=_STREAM)

    opcodes = [
        Counter(re.findall(_OPCODE, compiled_code.sass(launched)))
        for launched in (through_value, through_record)
    ]
    assert table.get().tolist() == held.get().tolist()
    assert opcodes[0] == opcodes[1]
