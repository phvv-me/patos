import annotationlib
import ast
import copy
import importlib
import itertools
import linecache
import operator
import pickle
import re
import textwrap
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import cache
from types import FunctionType, ModuleType, SimpleNamespace
from typing import NamedTuple, TypeAliasType

import cupy as cp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numba import types
from numba.core.errors import NumbaError, TypingError
from numba.core.target_extension import target_override
from numba.core.typing import templates
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda.target import CUDATypingContext
from numba.np.numpy_support import as_dtype, from_dtype

from patos.cuda.primitives import Bitmap, Filter, PairTable, bits
from patos.cuda.runtime import Workspace
from patos.cuda.scalars import ArrayOf
from patos.cuda.typed import (
    AnnotationError,
    Constant,
    Kernel,
    Matrix,
    Per,
    Struct,
    Vector,
    block_index,
    cuda,
    device,
    i16,
    i16x2,
    i32,
    i64,
    items,
    items_through,
    kernel,
    lane,
    number,
    ptx,
    thread_in_block,
    thread_index,
    u8,
    u8x4,
    u16,
    u32,
    u64,
    unsigned,
    warp_in_block,
    warp_index,
)
from patos.cuda.typed.arguments import Argument, argument
from patos.cuda.typed.arithmetic import meet, met, operated
from patos.cuda.typed.declarations import (
    Evaluated,
    IntLiteral,
    Kind,
    NamedValue,
    Signature,
    declared,
    named,
)

# Checks, rewrites and typing rules need only the `cuda` extra, as numba-cuda decorates lazily;
# what uploads or launches needs a device.
gpu = pytest.mark.skipif(not cp.cuda.is_available(), reason="uploads and launches need a GPU")

type Scalar = type[np.integer]
type Reading = Scalar | IntLiteral
# What `defined` fills a template field with: a scalar by its short name, or text as it is.
type Field = Scalar | str
type Decorator = Callable[[FunctionType], FunctionType | Kernel]
# A refusal a record raises, given a record of the `tables` fixture to start from.
type Refusal = Callable[[Tables], Struct | type | Argument | None]

_NAMES: dict[Scalar, str] = {
    np.int16: "i16", np.int32: "i32", np.int64: "i64", np.uint8: "u8", np.uint16: "u16",
    np.uint32: "u32", np.uint64: "u64",
}  # fmt: skip
# An array annotation's element: a scalar, or an open type naming a family of them.
_ELEMENTS: dict[type[np.number], str] = {
    **_NAMES, np.number: "number", np.unsignedinteger: "unsigned"
}  # fmt: skip
_NUMBA: dict[Kind, types.Type] = {kind: from_dtype(np.dtype(kind)) for kind in _NAMES}
_NUMBA[bool] = types.boolean
_DTYPES = [np.bool_, np.int8, *_NAMES, np.float32, np.float64]
scalars = st.sampled_from(list(_NAMES))
# numpy types a u64 met with a signed integer as a float, which no conversion round-trips.
not_u64 = st.sampled_from([kind for kind in _NAMES if kind is not np.uint64])
bounded = st.integers(-(2**40), 2**40)
# The edges of every scalar type, where a literal fits or does not, inside what Numba can type.
_EDGES = [
    edge
    for bit in (7, 8, 15, 16, 31, 32, 63, 64)
    for edge in (-(2**bit), 2**bit - 1, 2**bit)
    if -(2**63) <= edge < 2**64
]
literals = st.one_of(st.integers(-(2**63), 2**64 - 1), st.sampled_from(_EDGES)).map(IntLiteral)
operands = st.one_of(scalars, literals)
# A literal past int64 types as uint64 whatever it meets, which the shift reading leaves out.
int64_operands = st.one_of(scalars, literals.filter(lambda literal: literal.value < 2**63))
# The operators the C rules type, each family as `arithmetic` promises it.
_ARITHMETIC = (
    operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod,
    operator.and_, operator.or_, operator.xor,
    operator.iadd, operator.isub, operator.imul, operator.ifloordiv, operator.imod,
    operator.iand, operator.ior, operator.ixor,
)  # fmt: skip
_SHIFTS = (operator.lshift, operator.rshift, operator.ilshift, operator.irshift)
_COMPARISONS = (operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne)
_BINARY: dict[type[ast.operator], Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}  # fmt: skip
_SOURCES = itertools.count()


class Tables(Struct):
    """An open-addressed table: its slots, the mask a key hashes by, and a shift."""

    slots: Vector[u64]
    mask: u32
    shift: i32


class Lookup(Struct):
    """A record of a record, keys of any unsigned element, the rows a probe fills, and a flag."""

    tables: Tables
    keys: Vector[unsigned]
    found: Matrix[u64]
    exact: bool = True


class Window(Struct):
    """A record whose width is part of its device type, sizing a local array as a literal."""

    values: Vector[u64]
    width: Constant[int]

    @kernel
    def sums(self, out: Vector[u64]) -> None:
        """Each thread sums the `width` values from its own, staged through a local array."""
        start = cuda.grid(1)
        if start + self.width <= self.values.size:
            window = cuda.local.array(self.width, np.uint64)
            for offset in range(self.width):
                window[offset] = self.values[start + offset]
            for offset in range(self.width):
                out[start] += window[offset]


class Span(NamedTuple):
    """A `[start, end)` span."""

    start: i32
    end: i32


class Viewed(NamedTuple):
    """A window of bytes: the bytes and where it opens."""

    data: Vector[u8]
    base: i64


class Cell(NamedTuple):
    """An unsigned offset and a signed step."""

    offset: u32
    step: i32


class Stepped(NamedTuple):
    """An unsigned offset and a signed step that defaults to one."""

    offset: u32
    step: i32 = i32(1)


class Unfit(NamedTuple):
    """A value one field of which names no device type."""

    name: str


@kernel
def total(values: Vector[unsigned], sums: Vector[u64]) -> None:
    """Add every value into `sums[0]`."""
    item = cuda.grid(1)
    if item < values.size:
        cuda.atomic.add(sums, 0, u64(values[item]))


@kernel
def looked_up(
    table: PairTable, members: Filter, flags: Bitmap, keys: Vector[u64], found: Matrix[i64]
) -> None:
    """Each thread answers its key through the three hash records, flags by its low ten bits."""
    item = cuda.grid(1)
    if item < keys.size:
        key = keys[item]
        found[item, 0] = table.get(key)
        found[item, 1] = key in members
        found[item, 2] = (key & 1023) in flags


def noop(count: i32) -> None:
    """A kernel body the grid tests never launch."""


def member(self, out: Vector[u64]) -> None:
    """A kernel body taking `self`, which only a record class gives it."""


_NAMESPACE: dict[str, Callable | type | TypeAliasType | ModuleType] = {
    "u8": u8, "u16": u16, "u32": u32, "u64": u64, "i16": i16, "i32": i32, "i64": i64,
    "number": number, "unsigned": unsigned, "Vector": Vector, "Matrix": Matrix,
    "Struct": Struct, "Tables": Tables, "Per": Per,
    "cuda": cuda, "device": device, "kernel": kernel, "ptx": ptx, "np": np, "items": items,
    "items_through": items_through, "lane": lane, "thread_index": thread_index,
    "warp_index": warp_index, "block_index": block_index, "thread_in_block": thread_in_block,
    "warp_in_block": warp_in_block, "NamedTuple": NamedTuple, "Span": Span, "Viewed": Viewed,
    "Unfit": Unfit, "Cell": Cell, "Stepped": Stepped, "u8x4": u8x4, "i16x2": i16x2, "bits": bits,
}  # fmt: skip


def executed(template: str, **fields: Field) -> SimpleNamespace:
    """What `template` defines, run as a module of its own.

    The `{field}`s of the template are filled with `fields`, a scalar by its short name. It runs in
    a namespace of the patos types, the decorators and `Tables`, and its text is registered under a
    made-up file so that `inspect.getsource` finds it as it does a real module's. Device code that
    indexes a table by a tuple, takes the `len` of a record or calls a device function with the
    wrong array is spelled this way, since no type checker follows it.
    """
    filled = {
        name: _NAMES[kind] if not isinstance(kind, str) else kind for name, kind in fields.items()
    }
    text = textwrap.dedent(template).lstrip("\n").format(**filled)
    source = next(_SOURCES)
    filename = f"<patos-cuda-test-{source}>"
    linecache.cache[filename] = (len(text), None, text.splitlines(keepends=True), filename)
    namespace = _NAMESPACE | {"__name__": f"patos_cuda_test_{source}"}
    exec(compile(text, filename, "exec"), namespace)
    return SimpleNamespace(**namespace)


def defined(template: str, **fields: Field) -> FunctionType:
    """The function `f` that `template` defines, not decorated yet."""
    function = executed(template, **fields).f
    assert isinstance(function, FunctionType)
    return function


_DEVICE = executed(
    '''
    class Hashed(Struct):
        """A table read on the device through operators, a property and a method."""

        slots: Vector[u64]
        mask: u32
        shift: i32

        @device
        def __getitem__(self, index: i32) -> u64:
            return self.slots[index & self.mask]

        @device
        def __len__(self) -> i64:
            return len(self.slots)

        @device
        def __contains__(self, key: u64) -> bool:
            return self.slots[key & self.mask] == key

        @property
        @device
        def capacity(self) -> u32:
            return self.mask + 1

        @device
        def shifted(self, key: u64) -> u64:
            return key >> self.shift


    class Probe(Struct):
        """A hashed table, keys of any unsigned element, the rows a probe fills, and a flag."""

        hashed: Hashed
        keys: Vector[unsigned]
        found: Matrix[u64]
        exact: bool = True

        @kernel(per=Per.WARP, threads=64)
        def probe(self, base: u64) -> None:
            """Each warp's first lane fills its key's row through every member of the table."""
            item = cuda.grid(1) // 32
            if cuda.laneid == 0 and item < len(self.keys):
                key = u64(self.keys[item])
                self.found[item, 0] = self.hashed[item]
                self.found[item, 1] = len(self.hashed)
                self.found[item, 2] = key in self.hashed
                self.found[item, 3] = self.hashed.capacity
                self.found[item, 4] = self.hashed.shifted(key) + base
                self.found[item, 5] = self.exact


    @kernel
    def scatter(
        row: Vector[i64], table: Matrix[i16], small: u8, signed: i16, wide: u32, flag: bool
    ) -> None:
        """Write each scalar where the host reads it back, `signed` at the table's last corner."""
        row[0] = small
        row[1] = signed
        row[2] = wide
        row[3] = flag
        table[table.shape[0] - 1, table.shape[1] - 1] = signed


    @device
    def first(values: Vector[u8]) -> u8:
        return values[0]


    @kernel
    def delegate(values: Vector[i32], out: Vector[u8]) -> None:
        """Hand `values` to a device function declaring another element."""
        out[0] = first(values)
    '''
)


def annotation(text: str) -> Evaluated:
    """What `text` evaluates to as an annotation where the patos types are in scope."""
    return annotationlib.ForwardRef(text).evaluate(globals=_NAMESPACE)


def dispatched(function: FunctionType) -> CUDADispatcher:
    """`function` as `device` compiles it."""
    dispatcher = device(function)
    assert isinstance(dispatcher, CUDADispatcher)
    return dispatcher


def recorded(function: FunctionType) -> Signature:
    """What the decorator recorded of the annotations of the rewritten `function`."""
    return function.__dict__["device_signature"]


def host(template: str, **fields: Field) -> FunctionType:
    """The rewritten function `template` defines as plain Python, annotations converting."""
    return dispatched(defined(template, **fields)).py_func


def rejection(function: FunctionType, *, decorator: Decorator = device) -> str:
    """The `AnnotationError` message `decorator` raises for `function`."""
    with pytest.raises(AnnotationError) as caught:
        decorator(function)
    return str(caught.value)


def outcome[R: Struct](build: Callable[[], R]) -> R | set[str]:
    """The record `build` builds, or the fields its refusal names."""
    try:
        return build()
    except TypeError as error:
        return {problem.split()[0] for problem in str(error).split(": ", 1)[1].split("; ")}


def verdict(function: FunctionType) -> str:
    """The `AnnotationError` message `device` raises for `function`, empty when it accepts it."""
    try:
        device(function)
    except AnnotationError as error:
        return str(error)
    return ""


def ptx_of(launched: Kernel, *arguments, count: int = 1) -> str:
    """The PTX `launched` compiles to when launched over `count` items of `arguments`.

    A rename of its symbols leaves it as it is, and so does the environment pointer Numba declares
    for each function it links, used or not.
    """
    launched[count](*arguments)
    raw = launched.dispatcher.inspect_asm(next(iter(launched.dispatcher.overloads)))
    code = re.sub(r"//[^\n]*|\.(?:version|target|address_size)[^\n]*", "", raw)
    unused = {
        found[1]
        for found in re.finditer(r"\.common \.global [^;]*?([\w$]+);", code)
        if code.count(found[1]) == 1
    }
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    entry = re.search(r"\.entry ([\w$]+)\(", code)
    assert entry is not None
    return "\n".join(line for line in lines if not any(name in line for name in unused)).replace(
        entry[1], "kernel"
    )


def instructions(ptx: str) -> Counter[str]:
    """How many times each instruction of `ptx` appears, whatever its operands and order.

    A conjunction of five predicates is associated in another order on sm_121 for a named value
    read by attribute than for a plain tuple unpacked, with the same instructions.
    """
    return Counter(line.split()[0] for line in ptx.splitlines())


def named_rows(centers: Sequence[int], reach: int, data: np.ndarray) -> list[list[int]]:
    """The int64 rows of shape `[len(centers), 6]` that `_NAMED.named` writes, wrapped as `i32`.

    data: uint8 bytes of shape `[n]` that the last column reads at the start of each span.
    """
    rows = []
    for item, center in enumerate(centers):
        start, end = held([np.int32] * 2, [center - reach, center + reach])
        before, after = held([np.int32] * 2, [start - 1, end + 1])
        peeked = int(data[start]) if 0 <= start < len(data) else 0
        rows.append([start, end, wrapped(np.int32, after - before), 1, item + 5, peeked])
    return rows


def wrapped(kind: Scalar, value: int) -> int:
    """`value` as a C cast to `kind` leaves it, wrapping around."""
    bits = 8 * np.dtype(kind).itemsize
    value %= 1 << bits
    negative = np.issubdtype(kind, np.signedinteger) and value >> (bits - 1)
    return value - (1 << bits) if negative else value


def held(kinds: Sequence[Scalar], values: Sequence[int]) -> list[int]:
    """Each of `values` as a C cast to the matching one of `kinds` leaves it."""
    return [wrapped(kind, value) for kind, value in zip(kinds, values, strict=True)]


def typed_as(reading: Kind) -> types.Type:
    """The Numba type a reading stands for."""
    assert reading is not None
    return (
        types.IntegerLiteral(reading.value) if isinstance(reading, IntLiteral) else _NUMBA[reading]
    )


def is_negative_literal(reading: Reading) -> bool:
    return isinstance(reading, IntLiteral) and reading.value < 0


def decided(*, left: Reading, right: Reading) -> types.Type | None:
    """What `met` decides in the order Numba asks a rule that prefers literals.

    That order is the operands' literals first, then their plain types.
    """
    literal = met(typed_as(left), typed_as(right))
    return (
        literal
        if literal is not None
        else met(types.unliteral(typed_as(left)), types.unliteral(typed_as(right)))
    )


def resolved(
    context: CUDATypingContext, function: Callable, *arguments: types.Type
) -> templates.Signature:
    """The signature `function` takes on `arguments` on the CUDA target."""
    with target_override("cuda"):
        signature = context.resolve_function_type(function, arguments, {})
    assert signature is not None
    return signature


def probed(keys: np.ndarray, *, exact: bool, base: int) -> np.ndarray:
    """The uint64 rows of shape `[n, 6]` that `Probe.probe` writes for `keys`.

    keys: unsigned integers of shape `[n]`, looked up in slots `9 * i` under mask 7 and shift 3.
    """
    slots, index, wide = (
        np.arange(8, dtype=np.uint64) * 9,
        np.arange(len(keys)),
        keys.astype(np.uint64),
    )
    eight, flags = np.full(len(keys), 8), np.full(len(keys), exact)
    columns = [slots[index & 7], eight, slots[wide & 7] == wide, eight, (wide >> 3) + base, flags]
    return np.stack(columns, axis=1).astype(np.uint64)


@pytest.fixture(scope="module")
def context() -> CUDATypingContext:
    """A CUDA typing context carrying the rules the module registers."""
    typing = CUDATypingContext()
    typing.refresh()
    return typing


@pytest.fixture
def tables() -> Tables:
    """Slots `9 * i` under mask 7, so slot `i` holds the one key in it that hashes there."""
    return Tables(np.arange(8, dtype=np.uint64) * 9, 7, 3)


_UNSTANDING: dict[str, tuple[Decorator, str, str]] = {
    "unannotated parameter": (
        device,
        "def f(x) -> {kind}:\n    return x\n",
        "parameter `x` has no annotation",
    ),
    "every issue named": (device, "def f(x, y):\n    pass\n", "parameter `y` has no annotation"),
    "unannotated return": (
        device,
        "def f(x: {kind}):\n    return x\n",
        "the return has no annotation",
    ),
    "kernel returning": (
        kernel,
        "def f(x: {kind}) -> {kind}:\n    return x\n",
        "a kernel returns None",
    ),
    "value for None": (
        device,
        "def f(x: {kind}) -> None:\n    return x\n",
        "returns a value where None is declared",
    ),
    "nothing for a value": (
        device,
        "def f(x: {kind}) -> {kind}:\n    return\n",
        "returns nothing where {kind} is declared",
    ),
    "tuple whole": (
        device,
        "def f(x: {kind}) -> tuple[{kind}, {kind}]:\n    return x\n",
        "return the 2 elements so each converts",
    ),
    "no device type": (device, "def f(x: str) -> None:\n    pass\n", "`str` names no device type"),
    "array returned": (
        device,
        "def f(x: {kind}) -> Vector[{kind}]:\n    return x\n",
        "a device function returns no array",
    ),
    "record of an array returned": (
        device,
        "def f(x: {kind}) -> Tables:\n    return x\n",
        "a device function returns no array",
    ),
    "array local": (
        device,
        "def f(x: {kind}) -> None:\n    y: Vector[{kind}] = x\n",
        "`y` declares an array",
    ),
    "subscript declared": (
        device,
        "def f(x: Vector[{kind}]) -> None:\n    x[0]: {kind} = 1\n",
        "`x[0]` is no name to declare",
    ),
    "local twice": (
        device,
        "def f(x: {kind}) -> None:\n    y: {kind} = 0\n    y: {other} = 1\n",
        "`y` is declared {kind} already",
    ),
    "parameter again": (
        device,
        "def f(x: {kind}) -> None:\n    x: {other} = 1\n",
        "parameter `x` is declared again",
    ),
    "loop variable": (
        device,
        "def f(n: i32) -> None:\n    i: {kind} = 0\n    for i in range(n):\n        pass\n",
        "loop variable `i` is declared",
    ),
    "items outside a loop": (
        kernel,
        "def f(n: i32) -> None:\n    x = items(n)\n",
        "`items` is the iterable of a `for` loop",
    ),
    "items in a device function": (
        device,
        "def f(n: i32) -> None:\n    for i in items(n):\n        pass\n",
        "a device function takes its item as a parameter",
    ),
    "items of two bounds": (
        kernel,
        "def f(n: i32) -> None:\n    for i in items_through(n, n):\n        pass\n",
        "`items_through` takes one bound",
    ),
    "items of two variables": (
        kernel,
        "def f(n: i32) -> None:\n    for i, j in items(n):\n        pass\n",
        "an `items` loop names its item with one variable",
    ),
    "items with an else": (
        kernel,
        "def f(n: i32) -> None:\n    for i in items(n):\n        pass\n    else:\n        pass\n",
        "an `items` loop takes no `else`",
    ),
    "break without a stride": (
        kernel,
        "def f(n: i32) -> None:\n    for i in items(n):\n        break\n",
        "a kernel that does not stride has one item: `return` leaves it",
    ),
    "continue without a stride": (
        kernel,
        "def f(n: i32) -> None:\n    for i in items(n):\n        continue\n",
        "a kernel that does not stride has one item: `return` leaves it",
    ),
    "named value as a parameter of a kernel": (
        kernel,
        "def f(x: Span) -> None:\n    pass\n",
        "a launch passes no Span",
    ),
    "named value returned as a tuple": (
        device,
        "def f(x: i32) -> Span:\n    return x, x\n",
        "return `Span(...)` so the fields keep their names",
    ),
    "named value of an array returned": (
        device,
        "def f(x: Viewed) -> Viewed:\n    return x\n",
        "a device function returns no array",
    ),
    "field cast to its declaration": (
        device,
        "def f(x: i32) -> Span:\n    return Span(i32(x), x)\n",
        "`Span`'s parameter converts to i32",
    ),
    "field cast the operator already does": (
        device,
        "def f(x: Cell, y: i32) -> u32:\n    return x.offset + u32(y)\n",
        "`u32(y)` is a redundant cast: `x.offset` is u32, so the operator converts it",
    ),
    "named value of too many fields": (
        device,
        "def f(x: i32) -> Span:\n    return Span(x, x, x)\n",
        "`Span`: too many positional arguments",
    ),
    "named value of a missing field": (
        device,
        "def f(x: i32) -> Span:\n    return Span(x)\n",
        "`Span`: missing a required argument: 'end'",
    ),
    "named value of an unknown field": (
        device,
        "def f(x: i32) -> Span:\n    return Span(x, end=x, stop=x)\n",
        "`Span`: got an unexpected keyword argument 'stop'",
    ),
    "named value of a field given twice": (
        device,
        "def f(x: i32) -> Span:\n    return Span(x, start=x)\n",
        "`Span`: multiple values for argument 'start'",
    ),
    "named value of unpacked fields": (
        device,
        "def f(x: i32) -> Span:\n    pair = x, x\n    return Span(*pair)\n",
        "`Span`: takes its fields one by one, not unpacked",
    ),
    "named value with an undeclared field": (
        device,
        "def f(x: Unfit) -> None:\n    pass\n",
        "`Unfit` names no device type",
    ),
    "lanes as a parameter of a kernel": (
        kernel,
        "def f(x: u8x4) -> None:\n    pass\n",
        "a launch passes no u8x4; pass a u32 and convert it",
    ),
    "lanes cast to their own lanes": (
        device,
        "def f(x: u8x4) -> None:\n    y = u8x4(x)\n",
        "`u8x4(x)` is a redundant cast: `x` is already u8x4",
    ),
    "lanes the return annotation converts to": (
        device,
        "def f(x: {kind}) -> i16x2:\n    return i16x2(x)\n",
        "the return annotation converts to i16x2",
    ),
    "lanes a declaration converts to": (
        device,
        "def f(x: {kind}) -> None:\n    y: u8x4 = u8x4(x)\n",
        "the declaration of `y` converts to u8x4",
    ),
}


@pytest.mark.parametrize(("decorator", "template", "issue"), _UNSTANDING.values(), ids=_UNSTANDING)
@given(kind=scalars, other=scalars)
@settings(max_examples=20)
def test_an_annotation_that_cannot_stand_is_refused_line_by_line(
    *, decorator: Decorator, template: str, issue: str, kind: Scalar, other: Scalar
) -> None:
    """Every issue is named, each on a `file:line: function:` line of its own."""
    message = rejection(defined(template, kind=kind, other=other), decorator=decorator)

    assert issue.format(kind=_NAMES[kind]) in message
    assert all(re.match(r"<patos-cuda-test-\d+>:\d+: f: ", line) for line in message.splitlines())


@given(
    name=st.sampled_from(list(_ELEMENTS.values())),
    array=st.sampled_from(["Vector", "Matrix"]),
    shape=st.sampled_from(["int", "int, int", "3", ":", "float", "int, 2", "str, int"]),
)
def test_a_numeric_type_is_a_scalar_and_in_a_vector_or_a_matrix_an_array(
    *, name: str, array: str, shape: str
) -> None:
    """Called, a numeric type is its numpy scalar; in `Vector` or `Matrix`, an array of it."""
    element = next(kind for kind, short in _ELEMENTS.items() if short == name)

    assert annotation(f"{array}[{name}]") == ArrayOf(
        element, ("Vector", "Matrix").index(array) + 1
    )
    with pytest.raises(TypeError, match=re.escape(f"{name}[...] declares no array; write ")):
        annotation(f"{name}[{shape}]")
    with pytest.raises(TypeError, match=f"{array}\\[.*\\]: an array holds a numeric type"):
        annotation(f"{array}[{shape}]")
    assert [type(kind(7)) for kind in (u8, u16, u32, u64, i16, i32, i64)] == [
        np.uint8, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64
    ]  # fmt: skip


_REDUNDANT = {
    "value already of the type": (
        "def f(x: {kind}) -> None:\n    y = {kind}(x)\n",
        "`x` is already {kind}",
    ),
    "record field already of the type": (
        """
        class Header(Struct):
            mask: {kind}

        def f(header: Header) -> None:
            y = {kind}(header.mask)
        """,
        "`header.mask` is already {kind}",
    ),
    "declaration converts": (
        "def f(x: {other}) -> None:\n    y: {kind} = {kind}(x)\n",
        "the declaration of `y` converts to {kind}",
    ),
    "return annotation converts": (
        "def f(x: {other}) -> {kind}:\n    return {kind}(x)\n",
        "the return annotation converts to {kind}",
    ),
    "return annotation converts a tuple element": (
        "def f(x: {other}) -> tuple[{kind}, {other}]:\n    return {kind}(x), x\n",
        "the return annotation converts to {kind}",
    ),
    "return annotation converts a conditional branch": (
        "def f(c: bool, x: {other}) -> {kind}:\n    return {kind}(x) if c else x\n",
        "the return annotation converts to {kind}",
    ),
    "store into an array converts": (
        "def f(out: Vector[{kind}], x: {other}) -> None:\n    out[0] = {kind}(x)\n",
        "the store into `out` converts to {kind}",
    ),
    "callee's parameter converts": (
        """
        @device
        def callee(x: {kind}) -> {kind}:
            return x

        def f(y: {other}) -> None:
            z = callee({kind}(y))
        """,
        "`callee`'s parameter converts to {kind}",
    ),
}


@pytest.mark.parametrize(("template", "reason"), _REDUNDANT.values(), ids=_REDUNDANT)
@given(kind=scalars, other=scalars)
@settings(max_examples=20)
def test_a_cast_something_else_already_does_is_redundant(
    *, template: str, reason: str, kind: Scalar, other: Scalar
) -> None:
    """A cast to the type an annotation already converts to is rejected, naming that annotation."""
    message = rejection(defined(template, kind=kind, other=other))

    assert "is a redundant cast" in message
    assert reason.format(kind=_NAMES[kind]) in message


_STATEMENTS = {
    "right operand": "c = a + {kind}(b)",
    "left operand": "c = {kind}(b) * a",
    "comparison": "c = a < {kind}(b)",
    "min": "c = min(a, {kind}(b))",
    "augmented": "a += {kind}(b)",
}


@pytest.mark.parametrize("statement", _STATEMENTS.values(), ids=_STATEMENTS)
@pytest.mark.parametrize(
    ("kind", "other", "redundant"),
    [
        *[(kind, other, True) for kind, other in [
            (np.uint64, np.int32), (np.uint64, np.uint8), (np.uint64, np.uint64),
            (np.int64, np.int16), (np.int64, np.uint32), (np.uint32, np.int16),
            (np.uint32, np.uint8),
        ]],
        *[(kind, other, False) for kind, other in [
            (np.int32, np.int16), (np.int32, np.uint32), (np.uint32, np.uint64),
            (np.uint32, np.int64), (np.int64, np.uint64), (np.uint8, np.int16),
        ]],
    ],
)  # fmt: skip
def test_a_cast_is_redundant_exactly_where_the_operator_already_meets_in_its_type(
    *, statement: str, kind: Scalar, other: Scalar, redundant: bool
) -> None:
    """A cast of `b` to the type of `a` is redundant exactly where the operator meets them there.

    Everywhere else the cast changes the result, so it stays.
    """
    template = f"def f(a: {{kind}}, b: {{other}}) -> None:\n    {statement}\n"
    message = verdict(defined(template, kind=kind, other=other))

    assert (f"`a` is {_NAMES[kind]}, so the operator converts it" in message, bool(message)) == (
        redundant,
        redundant,
    )


@given(kind=scalars, other=scalars)
@settings(deadline=None)
def test_a_cast_is_rejected_only_where_numba_types_the_sum_as_the_cast(
    context: CUDATypingContext, *, kind: Scalar, other: Scalar
) -> None:
    """Soundness: a cast the decorator calls redundant never changes what Numba types."""
    template = "def f(a: {kind}, b: {other}) -> None:\n    c = a + {kind}(b)\n"
    message = verdict(defined(template, kind=kind, other=other))
    assume("so the operator converts it" in message)

    signature = resolved(context, operator.add, typed_as(kind), typed_as(other))

    assert signature.return_type == typed_as(kind)


def test_a_rebuilt_function_keeps_its_name_docstring_globals_and_signature() -> None:
    """A rebuilt device function keeps the original's name, docstring, globals and signature.

    Widening before a product repeats nothing, and reading the module's own globals lets a device
    function call one its module defines further down.
    """
    original = defined(
        '''
        def f(a: u32, b: u32, out: Vector[u64]) -> u64:
            """Widen before the product."""
            total: u64 = u64(a) * b
            out[0] = total
            return total
        '''
    )
    rebuilt = dispatched(original).py_func

    assert (rebuilt.__doc__, rebuilt.__qualname__) == ("Widen before the product.", "f")
    assert rebuilt.__globals__ is original.__globals__
    assert recorded(rebuilt) == Signature((np.uint32, np.uint32, ArrayOf(np.uint64, 1)), np.uint64)


def test_a_type_parameter_is_left_alone_and_a_type_alias_declares_its_type() -> None:
    """A generic parameter converts nothing, and `type Index = i32` converts like `i32`."""
    generic = host("def f[T](x: T) -> T:\n    return x\n")
    aliased = host("type Index = i32\n\ndef f(x: Index) -> Index:\n    return x\n")

    assert generic("unchanged") == "unchanged"
    assert type(aliased(np.int64(7))) is np.int32


_DECLARING = """
type Pair = tuple[{second}, {third}]

def f[T](x: {first}, pair: Pair, other: T, values: {array}[{element}]) -> {first}:
    return x
"""


@given(
    kinds=st.tuples(scalars, scalars, scalars),
    passed=st.tuples(scalars, scalars, scalars),
    element=st.sampled_from(list(_ELEMENTS)),
    dims=st.integers(1, 2),
    array=st.builds(
        types.Array, st.sampled_from(_DTYPES).map(lambda kind: from_dtype(np.dtype(kind))),
        st.integers(1, 2), st.sampled_from("CA"),
    ),
)  # fmt: skip
def test_a_device_function_compiles_at_the_types_its_parameters_declare(
    *,
    kinds: Sequence[Scalar],
    passed: Sequence[Scalar],
    element: type[np.number],
    dims: int,
    array: types.Array,
) -> None:
    """Numba compiles a device function at the parameter types it declares.

    Scalars and tuples, through a type alias too, take their declared types whatever the caller
    passes, and a type parameter keeps the type it is passed. An array keeps the layout it is
    passed when its element and dimensions are ones the parameter declares, and is refused with a
    typing error naming the parameter otherwise.
    """
    first, second, third = kinds
    function = dispatched(
        defined(
            _DECLARING,
            first=first,
            second=second,
            third=third,
            element=_ELEMENTS[element],
            array=("Vector", "Matrix")[dims - 1],
        )
    )
    admitted = array.ndim == dims and np.issubdtype(as_dtype(array.dtype), element)
    compiled: list[tuple[types.Type, ...]] = []
    with (
        pytest.MonkeyPatch.context() as patch,
        nullcontext()
        if admitted
        else pytest.raises(TypingError, match=re.escape(f"f's `values` receives {array} where")),
    ):
        patch.setattr(
            CUDADispatcher, "compile_device", lambda _, args, _returns=None: compiled.append(args)
        )
        function.compile_device(
            (_NUMBA[passed[0]], types.UniTuple(_NUMBA[passed[1]], 2), _NUMBA[passed[2]], array)
        )

    pair = types.BaseTuple.from_types([_NUMBA[second], _NUMBA[third]])
    assert compiled == ([(_NUMBA[first], pair, _NUMBA[passed[2]], array)] if admitted else [])


@given(kind=not_u64, flag=st.booleans(), a=bounded, b=bounded)
def test_a_declared_local_converts_every_assignment_to_its_declaration(
    *, kind: Scalar, flag: bool, a: int, b: int
) -> None:
    """Every assignment to a declared local converts to its declaration, wrapping at each step.

    Plain, augmented, conditional and unpacking assignments all convert, while an undeclared target
    of the same unpacking converts nothing.
    """
    template = """
    def f(flag: bool, a: i64, b: i64) -> tuple[i64, i64, i64]:
        cursor: {kind}
        cursor = a
        cursor = cursor + b
        cursor += b
        chosen: {kind} = a if flag else b
        other = 0
        cursor, other = cursor + a, b
        return cursor, chosen, other
    """
    cursor = wrapped(kind, wrapped(kind, wrapped(kind, a) + b) + b)
    expected = (wrapped(kind, cursor + a), wrapped(kind, a if flag else b), b)

    assert host(template, kind=kind)(flag, np.int64(a), np.int64(b)) == expected


@given(kinds=st.tuples(scalars, scalars, scalars), flag=st.booleans(), a=bounded, b=bounded)
def test_a_return_converts_element_by_element_and_branch_by_branch(
    *, kinds: tuple[Scalar, Scalar, Scalar], flag: bool, a: int, b: int
) -> None:
    """A return converts each element of a nested tuple to its annotated type.

    A conditional element converts in whichever branch runs.
    """
    template = """
    def f(flag: bool, a: i64, b: i64) -> tuple[{first}, tuple[{second}, {third}]]:
        return (a if flag else b), (b, a)
    """
    function = host(template, first=kinds[0], second=kinds[1], third=kinds[2])
    head, (middle, tail) = function(flag, np.int64(a), np.int64(b))

    assert (type(head), type(middle), type(tail)) == kinds
    assert [head, middle, tail] == held(kinds, [a if flag else b, b, a])


@given(left=operands, right=operands)
def test_the_static_meet_agrees_with_the_runtime_rule(*, left: Reading, right: Reading) -> None:
    """Wherever the runtime rule `met` decides, `meet` reads the same type, and only there."""
    runtime = met(typed_as(left), typed_as(right))
    static = meet(left, right)

    assert (static is None) if runtime is None else (typed_as(static) == runtime)


@given(
    other=st.one_of(operands, st.just(bool)),
    narrow=st.sampled_from([np.int16, np.uint8, np.uint16]),
)
def test_integers_narrower_than_32_bits_meet_as_i32(
    *, other: Reading | type[bool], narrow: Scalar
) -> None:
    """The C integer promotion holds statically and at runtime, on either side of the operator."""
    assert meet(narrow, other) is meet(np.int32, other)
    assert meet(other, narrow) is meet(other, np.int32)
    assert met(typed_as(narrow), typed_as(other)) == met(types.int32, typed_as(other))
    assert met(typed_as(other), typed_as(narrow)) == met(typed_as(other), types.int32)


@given(op=st.sampled_from(list(_BINARY)), left=int64_operands, right=int64_operands)
@settings(deadline=None)
def test_the_static_reading_agrees_with_the_registered_typing(
    context: CUDATypingContext, *, op: type[ast.operator], left: Reading, right: Reading
) -> None:
    """Whatever type `operated` reads for `left op right`, Numba's typing gives it too."""
    reading = operated(op(), left, right)
    assume(reading is not None)
    # `u32` meeting a literal is the gap `test_a_literal_takes_the_type_of_the_u32_it_meets` holds.
    assume(not (np.uint32 in (left, right) and IntLiteral in (type(left), type(right))))
    # A negative literal shifted by a u64 count: Numba meets it with the plain int64 in the u64.
    assume(
        not (op in (ast.LShift, ast.RShift) and is_negative_literal(left) and right is np.uint64)
    )

    signature = resolved(context, _BINARY[op], typed_as(left), typed_as(right))

    assert signature.return_type == typed_as(reading)


@given(op=st.sampled_from([*_ARITHMETIC, *_SHIFTS, *_COMPARISONS]), left=operands, right=operands)
@settings(deadline=None)
def test_the_registered_rules_type_operators_as_the_runtime_rule_decides(
    context: CUDATypingContext, *, op: Callable, left: Reading, right: Reading
) -> None:
    """Where `met` decides, an operator takes both operands in it and answers it or a bool."""
    meeting = decided(left=left, right=right)
    assume(meeting is not None and (op not in _SHIFTS or meeting in (types.int64, types.uint64)))

    signature = resolved(context, op, typed_as(left), typed_as(right))

    assert signature.args == (meeting, meeting)
    assert signature.return_type == (types.boolean if op in _COMPARISONS else meeting)


def test_a_literal_takes_the_type_of_the_u32_it_meets(context: CUDATypingContext) -> None:
    """Numba asks the rule about the literal before its plain `int64`, so `u32 + 1` stays u32."""
    signature = resolved(context, operator.add, types.uint32, types.IntegerLiteral(1))

    assert meet(np.uint32, IntLiteral(1)) is np.uint32
    assert signature.return_type == types.uint32


@pytest.mark.parametrize("module", ["warp", "hash"])
def test_the_ported_modules_pass_their_own_annotation_checks(*, module: str) -> None:
    """Importing a module decorates its device functions, which raises on a wrong annotation."""
    assert importlib.import_module(f"patos.cuda.primitives.{module}")


@given(
    kind=st.sampled_from([np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]),
    passed=scalars,
)
@settings(deadline=None)
def test_a_ptx_call_takes_the_stubs_declared_types_which_the_checks_read(
    context: CUDATypingContext, *, kind: Scalar, passed: Scalar
) -> None:
    """Numba types a PTX call at the stub's declared types, which the cast checks read as well.

    The caller's argument is cast to them, and casting the result to the type it returns is
    redundant.
    """
    caller = defined(
        """
        @ptx("mov.b64 $result, $value;")
        def g(value: {kind}) -> {kind}:
            ...

        def f(x: {kind}) -> i64:
            return {kind}(g(x))
        """,
        kind=kind,
    )
    # An intrinsic registers its typing as it is made, after the module's context was built.
    context.refresh()
    signature = resolved(context, caller.__globals__["g"], _NUMBA[passed])

    assert (signature.args, signature.return_type) == ((_NUMBA[kind],), _NUMBA[kind])
    assert f"`{_NAMES[kind]}(g(x))` is a redundant cast: `g(x)` is already" in rejection(caller)


@pytest.mark.parametrize(
    ("template", "error", "message"),
    [
        ("def f(value: u8) -> i32:\n    ...\n", AnnotationError, "16, 32 or 64 bits"),
        ("def f(value: Vector[i32]) -> i32:\n    ...\n", AnnotationError, "declares no scalar"),
        ("def f(other: i32) -> i32:\n    ...\n", KeyError, "value"),
    ],
)
def test_a_ptx_stub_declares_scalars_and_names_only_its_operands(
    *, template: str, error: type[Exception], message: str
) -> None:
    """A PTX stub that cannot be lowered is refused where it is defined.

    That is a stub without scalar types or with a byte operand, or a template naming an operand the
    stub lacks.
    """
    with pytest.raises(error, match=message):
        ptx("mov.b32 $result, $value;")(defined(template))


@given(
    per=st.sampled_from(Per),
    threads=st.sampled_from([32, 64, 128, 1024]),
    items=st.integers(0, 2**40),
)
def test_a_grid_gives_each_item_its_lanes_and_caps_a_striding_kernel(
    *, per: Per, threads: int, items: int
) -> None:
    """Each item gets the lanes it runs on with no block to spare, and striding caps the blocks.

    An item runs on a thread, a warp or a whole block.
    """
    plain, striding = (
        kernel(per=per, threads=threads, strided=strided)(noop) for strided in (False, True)
    )
    lanes = {Per.THREAD: 1, Per.WARP: 32, Per.BLOCK: threads}[per]
    blocks, block = plain.grid(items)
    cap = striding.grid(2**50)[0]

    assert block == striding.grid(items)[1] == threads
    assert (blocks - 1) * threads < items * lanes <= blocks * threads
    assert striding.grid(items)[0] == min(blocks, cap) and cap < plain.grid(2**50)[0]


@gpu
@given(
    dtype=st.sampled_from([np.uint64, np.uint8, np.int64, np.uint32, np.float64]),
    shape=st.sampled_from([(4,), (2, 4)]),
    strided=st.booleans(),
    mask=st.one_of(st.integers(-(2**33), 2**33), st.floats(-1e10, 1e10)),
    shift=st.one_of(st.integers(-(2**32), 2**32), st.floats(-1e10, 1e10)),
)
@settings(deadline=None, max_examples=50)
def test_a_record_names_every_field_it_refuses_at_once_and_converts_the_rest(
    *, dtype: type[np.generic], shape: tuple[int, ...], strided: bool, mask: float, shift: float
) -> None:
    """A record names every field it refuses at once, and builds when it refuses none.

    A scalar converts with an overflow check and never from a float, and a host array of the
    element and dimensions declared uploads contiguous, whatever its strides.
    """
    slots = np.zeros(shape, dtype)[..., :: 1 + strided]
    fits = {
        "slots": dtype is np.uint64 and len(shape) == 1,
        "mask": isinstance(mask, int) and 0 <= mask < 2**32,
        "shift": isinstance(shift, int) and -(2**31) <= shift < 2**31,
    }
    record = outcome(lambda: Tables(slots, mask, shift))

    refused = record if isinstance(record, set) else set()
    assert refused == {name for name, fit in fits.items() if not fit}
    assert isinstance(record, set) or (
        [type(record.mask), type(record.shift), record.mask, record.shift]
        == [np.uint32, np.int32, mask, shift]
        and cp.asarray(record.slots).flags.c_contiguous
    )


_REFUSALS: dict[str, tuple[Refusal, type[Exception], str]] = {
    "missing fields": (
        lambda _: Tables(),
        TypeError,
        "Tables: slots is missing; mask is missing; shift is missing",
    ),
    "too many values": (
        lambda tables: Tables(tables.slots, 1, 2, 3),
        TypeError,
        "Tables has 3 fields, 4 given",
    ),
    "a field given twice": (
        lambda tables: Tables(tables.slots, slots=tables.slots, mask=1, shift=1),
        TypeError,
        "Tables given slots twice",
    ),
    "an unknown field": (
        lambda tables: Tables(tables.slots, 1, 2, colour=3),
        TypeError,
        "Tables has no field colour",
    ),
    "a record of another class": (
        lambda tables: Lookup(3, tables.slots, cp.zeros((1, 6), np.uint64)),
        TypeError,
        "Lookup: tables is a int, not the Tables declared",
    ),
    "a scalar given an array": (
        lambda tables: Tables(tables.slots, tables.slots, 1),
        TypeError,
        "Tables: mask is a ndarray, not the u32 declared",
    ),
    "a replacement that does not fit": (
        lambda tables: copy.replace(tables, mask=-1),
        TypeError,
        "Tables: mask is -1, out of range of the u32 declared",
    ),
    "an assignment": (
        lambda tables: Tables.__setattr__(tables, "mask", 1),
        AttributeError,
        "Tables is frozen; copy.replace it",
    ),
    "a deletion": (
        lambda tables: Tables.__delattr__(tables, "mask"),
        AttributeError,
        "Tables is frozen",
    ),
    "a subclass": (
        lambda _: type("More", (Tables,), {"__annotations__": {"extra": u8}}),
        TypeError,
        "More extends a record, which takes no subclass",
    ),
    "a kernel taking self outside a record": (
        lambda _: type("Plain", (), {"probe": kernel(member)}),
        TypeError,
        "kernel probe takes `self`, which only a Struct gives it",
    ),
    "a constant of another kind": (
        lambda tables: Window(tables.slots, 2.5),
        TypeError,
        "Window: width is a float, not the int declared",
    ),
    "zeroing a field given no size": (
        lambda _: Tables.take(Workspace(cp), zeroed=("mask",), slots=2, mask=1, shift=0),
        TypeError,
        "Tables zeroes mask, given no size",
    ),
    "taking an open element": (
        lambda tables: Lookup.take(
            Workspace(cp), tables=tables, keys=2, found=cp.zeros((1, 6), np.uint64)
        ),
        TypeError,
        "Lookup.keys declares Vector[unsigned], which has no one dtype to take",
    ),
    "a strided device array, where the record marshals": (
        lambda tables: argument(Tables(tables.slots[::2], 1, 2)),
        TypeError,
        "Tables.slots is strided, not contiguous",
    ),
}


@gpu
@pytest.mark.parametrize(("refusal", "error", "message"), _REFUSALS.values(), ids=_REFUSALS)
def test_a_record_refuses_what_names_no_field_or_would_change_it(
    tables: Tables, *, refusal: Refusal, error: type[Exception], message: str
) -> None:
    """Records are built whole from their own fields, never changed in place, never extended."""
    with pytest.raises(error, match=re.escape(message)):
        refusal(tables)


@gpu
@given(mask=st.integers(0, 2**32 - 1), changed=st.integers(0, 2**32 - 1))
@settings(deadline=None, max_examples=25)
def test_a_record_rebuilds_through_its_validation_and_copies_as_its_arrays_do(
    *, mask: int, changed: int
) -> None:
    """`copy.replace` and `of` rebuild a record through its validation, sharing its arrays.

    A field `of`'s source lacks takes its default, a shallow copy shares the arrays, and a deep
    copy or a pickle copies them.
    """
    tables = Tables(np.arange(4, dtype=np.uint64), mask, 3)
    rebuilt = [
        copy.replace(tables, mask=changed),
        Tables.of({"slots": tables.slots, "mask": changed, "shift": 3, "stray": 0}),
        Tables.of(tables, mask=changed),
    ]
    copies = [copy.copy(tables), copy.deepcopy(tables), pickle.loads(pickle.dumps(tables))]
    lookup = Lookup.of(
        {"tables": tables, "keys": tables.slots, "found": cp.zeros((1, 6), np.uint64)}
    )

    assert [(record.slots is tables.slots, record.mask) for record in rebuilt] == [
        (True, changed)
    ] * 3
    assert [(record.slots is tables.slots, record.slots.get().tolist()) for record in copies] == [
        (True, [0, 1, 2, 3]), (False, [0, 1, 2, 3]), (False, [0, 1, 2, 3])
    ]  # fmt: skip
    assert {type(record.mask) for record in (*rebuilt, *copies)} == {np.uint32}
    assert (lookup.exact, lookup.tables is tables) == (True, True)
    assert repr(tables) == f"Tables(slots=u64[4], mask=np.uint32({mask}), shift=np.int32(3))"


@gpu
@given(size=st.integers(1, 64), zeroed=st.booleans(), junk=st.integers(1, 2**63 - 1))
@settings(deadline=None, max_examples=25)
def test_take_draws_a_sized_array_field_from_its_role_in_the_workspace(
    *, size: int, zeroed: bool, junk: int
) -> None:
    """A sized array field comes from the workspace under the record's role for that field.

    The role is the record's qualified name and the field's, the buffer is in the declared dtype
    and zeroed when asked, and every other field is given as it is.
    """
    workspace = Workspace(cp)
    role = f"{__name__}.Tables.slots"
    workspace.take(role, 64, np.uint64).fill(junk)
    tables = Tables.take(
        workspace, zeroed={"slots"} if zeroed else (), slots=size, mask=size, shift=-size
    )

    assert (list(workspace), cp.shares_memory(tables.slots, workspace.buffers[role])) == (
        [role],
        True,
    )
    assert tables.slots.get().tolist() == [0 if zeroed else junk] * size
    assert (tables.mask, tables.shift) == (size, -size)


@gpu
def test_device_members_answer_on_the_device_and_leave_the_host_class() -> None:
    """A record's kernel reads a record field through its operators, property and method.

    It compiles once per element its open array meets, whether a field is a CuPy array or any
    other device array, and the host class keeps none of the device members.
    """
    hashed = _DEVICE.Hashed(np.arange(8, dtype=np.uint64) * 9, 7, 3)
    keys = np.array([0, 9, 20, 63], np.uint8)
    narrow = _DEVICE.Probe(hashed, keys, cp.zeros((4, 6), np.uint64))
    wide = _DEVICE.Probe(
        hashed, keys.astype(np.uint32), cuda.device_array((4, 6), np.uint64), exact=False
    )
    for probe in (narrow, wide, narrow):
        probe.probe[len(keys)](5)

    assert np.array_equal(narrow.found.get(), probed(keys, exact=True, base=5))
    assert np.array_equal(cp.asarray(wide.found).get(), probed(keys, exact=False, base=5))
    assert len(_DEVICE.Probe.probe.compiled) == 2
    assert {"__getitem__", "__len__", "__contains__", "capacity", "shifted"}.isdisjoint(
        vars(_DEVICE.Hashed)
    )


@gpu
def test_a_constant_field_compiles_into_the_device_type_and_marshals_nothing() -> None:
    """Each width compiles its own kernel, which reads the width as a literal.

    The record passes the kernel its array alone, since the width is part of its type.
    """
    values = cp.arange(8, dtype=np.uint64)
    windows = [Window(values, 2), Window(values, 3), Window(values, 2)]
    sums = [cp.zeros(8, np.uint64) for _ in windows]
    for window, out in zip(windows, sums, strict=True):
        window.sums[8](out)

    assert [out.get().tolist() for out in sums] == [
        [sum(range(start, start + width)) if start + width <= 8 else 0 for start in range(8)]
        for width in (2, 3, 2)
    ]
    assert len(Window.sums.compiled) == 2
    assert all(argument(window)[1] == argument(values)[1] for window in windows)


@gpu
@given(
    scalars=st.tuples(
        st.integers(0, 2**8 - 1), st.integers(-(2**15), 2**15 - 1), st.integers(0, 2**32 - 1),
        st.booleans(),
    ),
    shape=st.tuples(st.integers(1, 4), st.integers(1, 4)),
    wrap=st.sampled_from([int, np.int64]),
)  # fmt: skip
@settings(deadline=None, max_examples=25)
def test_a_launch_converts_its_scalars_and_compiles_one_signature(
    *,
    scalars: tuple[int, int, int, bool],
    shape: tuple[int, int],
    wrap: Callable[[int], int | np.int64],
) -> None:
    """A launch converts Python or numpy integers to the declared scalars, compiling once.

    A launch over no items marshals and refuses nothing.
    """
    row, table = cp.zeros(4, np.int64), cp.zeros(shape, np.int16)
    _DEVICE.scatter[1](row, table, *map(wrap, scalars))
    _DEVICE.scatter[0](row.astype(np.int32), table, 1.5, 0, 0, True)

    assert row.get().tolist() == list(scalars)
    assert table.get().ravel().tolist() == [0] * (shape[0] * shape[1] - 1) + [scalars[1]]
    assert [kinds for kinds, _ in _DEVICE.scatter.compiled.values()] == [
        (types.Array(types.int64, 1, "C"), types.Array(types.int16, 2, "C"),
         types.uint8, types.int16, types.uint32, types.boolean)
    ]  # fmt: skip


def scattered(**changes: int | float | bool | cp.ndarray) -> None:
    """Launch `scatter` over one item with fitting arguments, `changes` replacing some of them.

    changes: arguments by parameter name; a fitting `row` is an int64 array of shape `[4]` and a
        fitting `table` an int16 array of shape `[2, 3]`.
    """
    fitting = {"row": cp.zeros(4, np.int64), "table": cp.zeros((2, 3), np.int16)}
    _DEVICE.scatter[1](
        *(fitting | {"small": 1, "signed": 1, "wide": 1, "flag": True} | changes).values()
    )


_UNDECLARED = {
    "element": (
        lambda: scattered(row=cp.zeros(4, np.int32)),
        TypeError,
        "scatter's `row` is array(int32, 1d, C), where Vector[i64] is declared",
    ),
    "dimensions": (
        lambda: scattered(row=cp.zeros((2, 2), np.int64)),
        TypeError,
        "scatter's `row` is array(int64, 2d, C), where Vector[i64] is declared",
    ),
    "strided": (
        lambda: scattered(row=cp.zeros(8, np.int64)[::2]),
        TypeError,
        "scatter's `row` is strided, not contiguous, where Vector[i64] is declared",
    ),
    "transposed": (
        lambda: scattered(table=cp.zeros((3, 2), np.int16).T),
        TypeError,
        "scatter's `table` is strided, not contiguous, where Matrix[i16] is declared",
    ),
    "open element": (
        lambda: total[4](cp.zeros(4, np.int32), cp.zeros(1, np.uint64)),
        TypeError,
        "total's `values` is array(int32, 1d, C), where",
    ),
    "overflow": (
        lambda: scattered(small=256),
        OverflowError,
        "scatter's `small` is 256, out of range of the u8 declared",
    ),
    "float": (
        lambda: scattered(signed=1.5),
        TypeError,
        "scatter's `signed` is a float, not the i16 declared",
    ),
    "device call": (
        lambda: _DEVICE.delegate[1](cp.zeros(2, np.int32), cp.zeros(1, np.uint8)),
        TypingError,
        "first's `values` receives array(int32, 1d, C) where Vector[u8] is declared",
    ),
}


@gpu
@pytest.mark.parametrize(("launch", "error", "message"), _UNDECLARED.values(), ids=_UNDECLARED)
def test_a_launch_refuses_what_a_parameter_does_not_declare(
    *, launch: Callable[[], None], error: type[Exception], message: str
) -> None:
    """A launch refuses an argument its parameter does not declare, naming the parameter.

    An array of another element, dimension count or layout is refused, as is a scalar its declared
    type cannot hold, and a device function refuses such an array at the call.
    """
    with pytest.raises(error, match=re.escape(message)):
        launch()


@gpu
def test_an_open_element_compiles_once_per_dtype_it_meets() -> None:
    """`Vector[unsigned]` takes a CuPy or any device array of each unsigned dtype it meets."""
    sums = cp.zeros(1, np.uint64)
    for values in (
        cp.arange(5, dtype=np.uint8),
        cuda.to_device(np.arange(5, dtype=np.uint16)),
        cp.arange(5, dtype=np.uint32),
        cp.arange(5, dtype=np.uint8),
    ):
        total[5](values, sums)

    assert (sums.get().tolist(), len(total.compiled)) == ([40], 3)


@gpu
@given(
    entries=st.dictionaries(st.integers(0, 2**64 - 2), st.integers(0, 2**63 - 1), min_size=1),
    strays=st.lists(st.integers(0, 2**64 - 2), max_size=8),
)
@settings(deadline=None, max_examples=25)
def test_the_hash_records_answer_on_the_device_what_the_host_built_them_from(
    *, entries: dict[int, int], strays: list[int]
) -> None:
    """Each record answers on the device what the host built it from.

    A pair table answers each key's payload and -1 for a stray, a filter holds every value it was
    built from, and a bitmap holds exactly the flags it packed.
    """
    keys = np.array([*entries, *strays], dtype=np.uint64)
    flags = np.zeros(1024, dtype=np.bool_)
    flags[[key & 1023 for key in entries]] = True
    found = cp.zeros((len(keys), 3), np.int64)
    table, members = PairTable.build(entries), Filter.build(entries)
    looked_up[len(keys)](table, members, Bitmap.pack(flags), cp.asarray(keys), found)

    rows = found.get()
    assert rows[:, 0].tolist() == [entries.get(key, -1) for key in keys.tolist()]
    assert rows[: len(entries), 1].all()
    assert rows[:, 2].tolist() == [int(flags[key & 1023]) for key in keys.tolist()]


@device
def always() -> bool:
    return True


@device
def first_lane() -> bool:
    return lane() == 0


@device
def first_thread() -> bool:
    return thread_in_block() == 0


_LEADERS = {Per.THREAD: always, Per.WARP: first_lane, Per.BLOCK: first_thread}


@cache
def counting(per: Per, *, strided: bool) -> Kernel:
    """A kernel counting each item its `items` loop visits below `count`, once for the item."""
    leads = _LEADERS[per]

    @kernel(per=per, threads=64, strided=strided)
    def visit(visits: Vector[i32], count: i32) -> None:
        for item in items(count):
            if leads():
                visits[item] += 1

    return visit


_EXITS = executed(
    """
    @kernel(strided=True, threads=32)
    def even_below(visits: Vector[i32], live: Vector[i32], cap: i32) -> None:
        for item in items_through(live[0]):
            if item % 2 == 1:
                continue
            if item >= cap:
                break
            visits[item] += 1


    @kernel(strided=True, threads=1)
    def growing(visits: Vector[i32], live: Vector[i32], cap: i32) -> None:
        for item in items(live[0]):
            visits[item] += 1
            if live[0] < cap:
                live[0] += 1
    """
)

_LOOPS = executed(
    """
    @kernel(strided=True)
    def thread_items(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        for item in items(live[0]):
            out[item] = values[item] * 2


    @kernel(strided=True)
    def thread_hand(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        item = cuda.grid(1)
        stride = cuda.gridsize(1)
        while item < live[0]:
            out[item] = values[item] * 2
            item += stride


    @kernel(per=Per.WARP, strided=True)
    def warp_items(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        for item in items(live[0]):
            if lane() == 0:
                out[item] = values[item] * 2


    @kernel(per=Per.WARP, strided=True)
    def warp_hand(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        item = cuda.grid(1) // 32
        stride = cuda.gridsize(1) // 32
        while item < live[0]:
            if cuda.laneid == 0:
                out[item] = values[item] * 2
            item += stride


    @kernel(per=Per.BLOCK, strided=True)
    def block_items(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        for item in items(live[0]):
            if thread_in_block() == 0:
                out[item] = values[item] * 2


    @kernel(per=Per.BLOCK, strided=True)
    def block_hand(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        item = cuda.blockIdx.x
        stride = cuda.gridDim.x
        while item < live[0]:
            if cuda.threadIdx.x == 0:
                out[item] = values[item] * 2
            item += stride


    @kernel
    def guard_items(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        for item in items(live[0]):
            out[item] = values[item] * 2


    @kernel
    def guard_hand(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        item = cuda.grid(1)
        if item < live[0]:
            out[item] = values[item] * 2


    @kernel(strided=True)
    def through_items(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        for item in items_through(live[0]):
            out[item] = values[item] * 2


    @kernel(strided=True)
    def through_hand(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        item = cuda.grid(1)
        stride = cuda.gridsize(1)
        while item <= live[0]:
            out[item] = values[item] * 2
            item += stride


    @kernel(strided=True)
    def continue_items(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        for item in items(live[0]):
            if values[item] & 1:
                continue
            out[item] = values[item] * 2


    @kernel(strided=True)
    def continue_hand(values: Vector[i32], out: Vector[i32], live: Vector[i32]) -> None:
        item = cuda.grid(1)
        stride = cuda.gridsize(1)
        while item < live[0]:
            if values[item] & 1:
                item += stride
                continue
            out[item] = values[item] * 2
            item += stride
    """
)

# Each identity helper's call, the expression it names, its value for thread `t` of 128-thread
# blocks and its type.
_IDENTITIES: dict[str, tuple[str, str, Callable[[int], int], type[np.integer]]] = {
    "thread_index": ("thread_index()", "cuda.grid(1)", lambda t: t, np.int64),
    "warp_index": ("warp_index()", "cuda.grid(1) // 32", lambda t: t // 32, np.int64),
    "block_index": ("block_index()", "cuda.blockIdx.x", lambda t: t // 128, np.int32),
    "lane": ("lane()", "cuda.laneid", lambda t: t % 32, np.int32),
    "thread_in_block": ("thread_in_block()", "cuda.threadIdx.x", lambda t: t % 128, np.int32),
    "warp_in_block": (
        "warp_in_block()", "cuda.threadIdx.x // 32", lambda t: t % 128 // 32, np.int32
    ),
}  # fmt: skip

_TILING = executed(
    """
    class Payload(Struct):
        starts: Vector[i32]
        ends: Vector[i32]


    class Tiling(Struct):
        padding: i32

        @device
        def bounds(self, payload: Payload) -> tuple[i32, i32]:
            tile = warp_index()
            return payload.starts[tile] - self.padding, payload.ends[tile] + self.padding

        @device
        def bounds_by_hand(
            self, starts: Vector[i32], ends: Vector[i32], tile: i64
        ) -> tuple[i32, i32]:
            return starts[tile] - self.padding, ends[tile] + self.padding

        @kernel(per=Per.WARP, threads=64)
        def measure(self, payload: Payload, out: Matrix[i32]) -> None:
            tile = warp_index()
            if tile < out.shape[0] and lane() == 0:
                first, last = self.bounds(payload)
                out[tile, 0] = first
                out[tile, 1] = last

        @kernel(per=Per.WARP, threads=64)
        def measure_by_hand(self, payload: Payload, out: Matrix[i32]) -> None:
            tile = cuda.grid(1) // 32
            if tile < out.shape[0] and cuda.laneid == 0:
                first, last = self.bounds_by_hand(payload.starts, payload.ends, tile)
                out[tile, 0] = first
                out[tile, 1] = last
    """
)

_NAMED = executed(
    """
    class Window(NamedTuple):
        base: u64
        capacity: u32


    @device
    def around(at: i64, reach: i64) -> Span:
        return Span(at - reach, at + reach)


    @device
    def around_by_name(at: i64, reach: i64) -> Span:
        return Span(end=at + reach, start=at - reach)


    @device
    def width(span: Span) -> i32:
        return span.end - span.start


    @device
    def widened(span: Span, by: i32) -> Span:
        start, end = span
        return Span(start - by, end + by)


    @device
    def window(base: i64, capacity: i64) -> Window:
        return Window(base, capacity)


    @device
    def room(held: Window) -> u64:
        return held.base + held.capacity


    @device
    def peek(view: Viewed, index: i64) -> u8:
        return view.data[view.base + index]


    @kernel
    def named(centers: Vector[i64], reach: i64, data: Vector[u8], out: Matrix[i64]) -> None:
        item = cuda.grid(1)
        if item < centers.size:
            span: Span = around(centers[item], reach)
            wide = widened(span, 1)
            out[item, 0] = span.start
            out[item, 1] = span.end
            out[item, 2] = width(wide)
            out[item, 3] = around_by_name(centers[item], reach)[0] == span.start
            out[item, 4] = room(window(item, 5))
            if 0 <= span.start and span.start < data.size:
                out[item, 5] = peek(Viewed(data, span.start), 0)


    @kernel
    def viewed(data: Vector[unsigned], out: Vector[i64]) -> None:
        out[0] = peek(Viewed(data, 1), 1)


    @cuda.jit(device=True)
    def misfit_span():
        return Span(np.int64(1), np.int64(2))


    @kernel
    def misfit(out: Vector[i32]) -> None:
        out[0] = width(misfit_span())
    """
)

_CONTEXTS = executed(
    """
    type Plain = tuple[i32, i32, i32, i32, i32]


    class Context(NamedTuple):
        cursor: i32
        state: i32
        accepted: i32
        scalar_start: i32
        gap: i32


    @device
    def opened_plain(at: i32) -> Plain:
        return at, 0, at, at, at


    @device
    def opened(at: i32) -> Context:
        return Context(at, 0, at, at, at)


    @device
    def stepped_plain(context: Plain, byte: i32) -> Plain:
        cursor, state, accepted, scalar_start, gap = context
        if byte == 10:
            return cursor + 1, 0, cursor + 1, cursor + 1, cursor + 1
        return cursor + 1, state ^ byte, accepted, scalar_start, gap


    @device
    def stepped(context: Context, byte: i32) -> Context:
        cursor, state, accepted, scalar_start, gap = context
        if byte == 10:
            return Context(cursor + 1, 0, cursor + 1, cursor + 1, cursor + 1)
        return Context(cursor + 1, state ^ byte, accepted, scalar_start, gap)


    @device
    def same_plain(context: Plain, entry: Plain) -> bool:
        cursor, state, accepted, scalar_start, gap = context
        entry_cursor, entry_state, entry_accepted, entry_scalar_start, entry_gap = entry
        return (
            state == entry_state
            and accepted == entry_accepted
            and scalar_start == entry_scalar_start
            and accepted > cursor
            and entry_accepted > entry_cursor
        )


    @device
    def same(context: Context, entry: Context) -> bool:
        return (
            context.state == entry.state
            and context.accepted == entry.accepted
            and context.scalar_start == entry.scalar_start
            and context.accepted > context.cursor
            and entry.accepted > entry.cursor
        )


    @kernel
    def walk_plain(chars: Vector[u8], out: Vector[i32]) -> None:
        item = cuda.grid(1)
        if item < out.size:
            context = opened_plain(item)
            entry = opened_plain(item + 1)
            for index in range(chars.size):
                context = stepped_plain(context, chars[index])
                entry = stepped_plain(entry, chars[index] ^ 1)
            out[item] = same_plain(context, entry)


    @kernel
    def walk_named(chars: Vector[u8], out: Vector[i32]) -> None:
        item = cuda.grid(1)
        if item < out.size:
            context = opened(item)
            entry = opened(item + 1)
            for index in range(chars.size):
                context = stepped(context, chars[index])
                entry = stepped(entry, chars[index] ^ 1)
            out[item] = same(context, entry)
    """
)


def test_a_named_tuple_declares_its_fields_and_one_with_an_undeclared_field_declares_nothing() -> (
    None
):
    """A named value is declared by its class, field by field, and spelled by the class's name."""
    span = declared(annotation("Span"))

    assert span == NamedValue(Span, (("start", np.int32), ("end", np.int32)))
    assert isinstance(span, NamedValue) and span.kinds() == (np.int32, np.int32)
    assert named(span) == "Span"
    assert declared(annotation("Unfit")) is None
    assert isinstance(viewed := declared(annotation("Viewed")), NamedValue)
    assert viewed.field("data") == ArrayOf(np.uint8, 1)


def test_a_record_holds_no_named_value() -> None:
    """A record marshals scalars, arrays and records, so a field naming a value fails at once."""

    class Holds(Struct):
        span: Span

    class Packs(Struct):
        lanes: u8x4

    with pytest.raises(TypeError, match=r"Holds\.span is a named value"):
        Holds.declarations()
    with pytest.raises(TypeError, match=r"Packs\.lanes is a named value or lanes"):
        Packs.declarations()


@pytest.mark.parametrize("name", _IDENTITIES)
def test_an_identity_helper_returns_the_type_numba_gives_the_expression_it_names(
    name: str,
) -> None:
    """The index of a thread in the grid is 64 bits wide, every other identity 32."""
    assert recorded(globals()[name].py_func) == Signature((), _IDENTITIES[name][3])


@pytest.mark.parametrize("name", _IDENTITIES)
@gpu
def test_an_identity_helper_gives_the_value_of_the_expression_it_names_in_the_same_ptx(
    name: str,
) -> None:
    """Called in a kernel, a helper is the expression written there: its value and its PTX."""
    call, expression, expected, _ = _IDENTITIES[name]
    kernels = [
        executed(
            f"""
            @kernel(threads=128)
            def f(out: Vector[i64]) -> None:
                out[cuda.grid(1)] = {stored}
            """
        ).f
        for stored in (call, expression)
    ]
    outs = [cp.zeros(384, np.int64) for _ in kernels]
    ptx = [ptx_of(launched, out, count=384) for launched, out in zip(kernels, outs, strict=True)]
    wanted = [expected(t) for t in range(384)]

    assert ptx[0] == ptx[1]
    assert outs[0].get().tolist() == outs[1].get().tolist() == wanted


@gpu
@pytest.mark.parametrize("shape", ["thread", "warp", "block", "guard", "through", "continue"])
def test_an_items_loop_compiles_to_the_loop_a_kernel_writes_by_hand(*, shape: str) -> None:
    """The PTX of an items loop is the hand-written loop's.

    The bound is reloaded each pass and the stride held in a register, and a kernel that does not
    stride has only its guard.
    """
    values, live = cp.arange(64, dtype=np.int32), cp.array([40], np.int32)
    outs = [cp.zeros(64, np.int32) for _ in range(2)]
    ptx = [
        ptx_of(getattr(_LOOPS, f"{shape}_{kind}"), values, out, live)
        for kind, out in zip(("items", "hand"), outs, strict=True)
    ]

    assert ptx[0] == ptx[1]
    assert outs[0].get().tolist() == outs[1].get().tolist()
    assert outs[0].any()


@gpu
@given(
    per=st.sampled_from(Per),
    strided=st.booleans(),
    count=st.integers(0, 150),
    launched=st.integers(1, 150),
)
@settings(deadline=None, max_examples=40)
def test_an_items_loop_visits_each_item_of_the_kernel_below_its_bound_once(
    *, per: Per, strided: bool, count: int, launched: int
) -> None:
    """An items loop visits each item below its bound once.

    A kernel that strides reaches every one however few items it launched, any other kernel its
    own, which a launch rounds up to whole blocks.
    """
    visit = counting(per, strided=strided)
    blocks, threads = visit.grid(launched)
    held = blocks * threads // per.lanes(threads)
    visits = cp.zeros(max(count, held) + 1, np.int32)
    visit[launched](visits, count)

    reached = count if strided else min(count, held)
    assert visits.get().tolist() == [1] * reached + [0] * (len(visits) - reached)


@gpu
@given(live=st.integers(0, 120), cap=st.integers(0, 130))
@settings(deadline=None, max_examples=30)
def test_an_items_loop_skips_by_continue_leaves_by_break_and_may_end_inclusive(
    *, live: int, cap: int
) -> None:
    """`continue` goes on to the thread's next item and `break` ends its loop."""
    visits, bound = cp.zeros(140, np.int32), cp.array([live], np.int32)
    _EXITS.even_below[3](visits, bound, cap)

    assert visits.get().tolist() == [int(i % 2 == 0 and i < cap and i <= live) for i in range(140)]


@gpu
@given(live=st.integers(0, 12), cap=st.integers(0, 20))
@settings(deadline=None, max_examples=30)
def test_an_items_loop_reads_its_bound_again_each_pass_as_a_while_loop_does(
    *, live: int, cap: int
) -> None:
    """A device bound that the loop itself grows is followed to where it stops."""
    bound, seen, reached = live, [], 0
    while reached < bound:
        seen.append(reached)
        bound += bound < cap
        reached += 1
    visits, held = cp.zeros(24, np.int32), cp.array([live], np.int32)
    _EXITS.growing[1](visits, held, cap)

    assert visits.get().tolist() == [int(i in seen) for i in range(24)]
    assert held.get().tolist() == [bound]


@gpu
@given(
    padding=st.integers(-5, 5),
    spans=st.lists(st.tuples(st.integers(0, 90), st.integers(0, 10)), min_size=1, max_size=8),
)
@settings(deadline=None, max_examples=20)
def test_a_device_method_takes_a_record_and_an_identity_helper_in_the_ptx_of_the_hand_written_one(
    *, padding: int, spans: Sequence[tuple[int, int]]
) -> None:
    """`tiling.bounds(payload)` reads the fields of `payload` and the warp's own index.

    It compiles to the PTX of `bounds_by_hand(payload.starts, payload.ends, tile)`, which is
    passed them.
    """
    starts = np.array([start for start, _ in spans], np.int32)
    payload = _TILING.Payload(starts, starts + np.array([w for _, w in spans], np.int32))
    tiling = _TILING.Tiling(padding)
    outs = [cp.zeros((len(spans), 2), np.int32) for _ in range(2)]
    kernels = [_TILING.Tiling.measure, _TILING.Tiling.measure_by_hand]
    ptx = [
        ptx_of(k, tiling, payload, out, count=len(spans))
        for k, out in zip(kernels, outs, strict=True)
    ]
    wanted = np.stack([payload.starts.get() - padding, payload.ends.get() + padding], axis=1)

    assert ptx[0] == ptx[1]
    assert all(np.array_equal(out.get(), wanted) for out in outs)


@gpu
@given(
    centers=st.lists(st.integers(-(2**40), 2**40), min_size=1, max_size=8),
    reach=st.integers(0, 2**33),
)
@settings(deadline=None, max_examples=25)
def test_a_named_value_converts_its_fields_and_is_read_passed_and_returned_on_the_device(
    *, centers: list[int], reach: int
) -> None:
    """A named value converts its fields when built and travels the device.

    A construction converts each field to its declaration, positionally or by name, and the value
    is read by attribute, by index and by unpacking, passed and returned.
    """
    data = (np.arange(60) * 3 + 7).astype(np.uint8)
    out = cp.zeros((len(centers), 6), np.int64)
    centers_on_device = cp.asarray(np.array(centers, np.int64))
    _NAMED.named[len(centers)](centers_on_device, reach, cp.asarray(data), out)

    assert out.get().tolist() == named_rows(centers, reach, data)


@gpu
def test_a_named_value_of_other_fields_than_declared_is_refused_where_it_is_passed() -> None:
    """Numba converts one named tuple to no other, so a call is refused, never converted."""
    with pytest.raises(TypingError, match=r"`span` receives .* where Span is declared"):
        _NAMED.misfit[1](cp.zeros(1, np.int32))
    data, out = cp.arange(4, dtype=np.uint8), cp.zeros(1, np.int64)
    _NAMED.viewed[1](data, out)
    assert out.get().tolist() == [2]
    with pytest.raises(TypingError, match=r"`view` receives .* where Viewed is declared"):
        _NAMED.viewed[1](data.astype(np.uint16), out)


@gpu
def test_a_named_context_compiles_to_the_ptx_of_the_plain_tuple_it_replaces() -> None:
    """A named `Context` compiles to the PTX of the plain tuple it replaces.

    Five fields are walked through two device functions, read by attribute or by unpacking and
    built by their class, as the five-tuple is.
    """
    chars = cp.asarray(np.frombuffer(b"abc\ndef\n\nxyz" * 3, np.uint8))
    outs = [cp.zeros(16, np.int32) for _ in range(2)]
    ptx = [
        ptx_of(launched, chars, out)
        for launched, out in zip((_CONTEXTS.walk_plain, _CONTEXTS.walk_named), outs, strict=True)
    ]

    assert instructions(ptx[0]) == instructions(ptx[1])
    assert outs[0].get().tolist() == outs[1].get().tolist()


_MEMBERS = executed(
    """
    class Bounds(NamedTuple):
        start: i32
        end: i32

        @device
        def length(self) -> i32:
            return self.end - self.start

        @device
        def holds(self, at: i32) -> bool:
            return self.start <= at and at < self.end

        @property
        @device
        def empty(self) -> bool:
            return self.end <= self.start

        @device
        def widened(self, by: i32) -> Bounds:
            return Bounds(self.start - by, self.end + by)


    class Room(NamedTuple):
        base: u64
        capacity: u32

        @device
        def reserve(self, count: u32) -> Room:
            return Room(self.base + count, self.capacity - count)

        @property
        @device
        def end(self) -> u64:
            return self.base + self.capacity


    class Pair(NamedTuple):
        end: i32
        start: i32

        @device
        def length(self) -> i32:
            return self.start * 100 + self.end


    @device
    def length(start: i32, end: i32) -> i32:
        return end - start


    @device
    def pair_length(start: i32, end: i32) -> i32:
        return start * 100 + end


    @device
    def holds(start: i32, end: i32, at: i32) -> bool:
        return start <= at and at < end


    @device
    def empty(start: i32, end: i32) -> bool:
        return end <= start


    @device
    def widened(start: i32, end: i32, by: i32) -> tuple[i32, i32]:
        return start - by, end + by


    @device
    def reserve(base: u64, capacity: u32, count: u32) -> tuple[u64, u32]:
        return base + count, capacity - count


    @device
    def end(base: u64, capacity: u32) -> u64:
        return base + capacity


    @kernel
    def methods(starts: Vector[i32], ends: Vector[i32], at: i32, out: Matrix[i64]) -> None:
        for item in items(starts.size):
            bounds = Bounds(starts[item], ends[item])
            room = Room(item, 64).reserve(starts[item] & 7)
            out[item, 0] = bounds.length()
            out[item, 1] = bounds.holds(at)
            out[item, 2] = bounds.empty
            out[item, 3] = bounds.widened(2).length()
            out[item, 4] = room.base
            out[item, 5] = room.end
            out[item, 6] = Pair(item, 7).length()


    @kernel
    def loose(starts: Vector[i32], ends: Vector[i32], at: i32, out: Matrix[i64]) -> None:
        for item in items(starts.size):
            start, end_ = starts[item], ends[item]
            base, capacity = reserve(item, 64, starts[item] & 7)
            low, high = widened(start, end_, 2)
            out[item, 0] = length(start, end_)
            out[item, 1] = holds(start, end_, at)
            out[item, 2] = empty(start, end_)
            out[item, 3] = length(low, high)
            out[item, 4] = base
            out[item, 5] = end(base, capacity)
            out[item, 6] = pair_length(7, item)


    @kernel
    def misspelled(out: Vector[i32]) -> None:
        out[0] = Bounds(1, 2).measure()


    @kernel
    def unheld(out: Vector[i32]) -> None:
        out[0] = Bounds(1, 2).holds()


    @kernel
    def overheld(out: Vector[i32]) -> None:
        out[0] = Bounds(1, 2).holds(1, 2)


    class Sized(NamedTuple):
        size: i32

        @device
        def __len__(self) -> i32:
            return self.size
    """
)


def members_rows(spans: Sequence[tuple[int, int]], at: int) -> list[list[int]]:
    """The int64 rows of shape `[n, 7]` that `_MEMBERS.methods` writes for spans and `at`."""
    rows = []
    for item, (start, end) in enumerate(spans):
        taken = start & 7
        base, capacity = item + taken, 64 - taken
        rows.append(
            [
                end - start,
                start <= at < end,
                end <= start,
                end - start + 4,
                base,
                base + capacity,
                700 + item,
            ]  # fmt: skip
        )
    return [[int(value) for value in row] for row in rows]


@gpu
@given(
    spans=st.lists(
        st.tuples(st.integers(-1000, 1000), st.integers(-1000, 1000)), min_size=1, max_size=8
    ),
    at=st.integers(-1000, 1000),
)
@settings(deadline=None, max_examples=15)
def test_a_named_value_has_methods_and_attributes_in_the_ptx_of_the_loose_scalars(
    *, spans: list[tuple[int, int]], at: int
) -> None:
    """Methods, attributes and a method returning a named value are device functions of fields.

    `Bounds(start, end).length()`, `bounds.holds(at)`, `bounds.empty` and `room.reserve(count)`
    compile to what device functions of the loose fields do, instruction for instruction, and
    `Pair(...).length()` answers for its own class beside `Bounds`'s, as `Room.end` does beside
    the field `Bounds.end`.
    """
    starts, ends = (cp.asarray(np.array(column, np.int32)) for column in zip(*spans, strict=True))
    outs = [cp.zeros((len(spans), 7), np.int64) for _ in range(2)]
    ptx = [
        ptx_of(launched, starts, ends, at, out, count=len(spans))
        for launched, out in zip((_MEMBERS.methods, _MEMBERS.loose), outs, strict=True)
    ]

    wanted = members_rows(spans, at)
    assert instructions(ptx[0]) == instructions(ptx[1])
    assert outs[0].get().tolist() == outs[1].get().tolist() == wanted


@gpu
@pytest.mark.parametrize(
    ("launched", "refusal"),
    [
        ("misspelled", r"Unknown attribute 'measure' of type Bounds"),
        ("unheld", r"Bounds\.holds: missing a required argument: 'at'"),
        ("overheld", r"Bounds\.holds: too many positional arguments"),
    ],
)
def test_a_call_of_a_method_its_named_value_lacks_or_with_other_arguments_is_refused(
    *, launched: str, refusal: str
) -> None:
    """numba-cuda raises its own error for an unknown attribute and patos Numba's for the rest."""
    with pytest.raises(NumbaError, match=refusal):
        getattr(_MEMBERS, launched)[1](cp.zeros(1, np.int32))


def test_a_named_value_keeps_the_operators_of_a_tuple() -> None:
    """An operator defined on a named value is refused when device code first names it."""

    def sized(value: _MEMBERS.Sized) -> i32:
        return value.size

    with pytest.raises(TypeError, match=r"Sized\.__len__: a named value keeps the operators"):
        device(sized)


_LANES = executed(
    """
    @device
    def relabelled(word: u32) -> i16x2:
        return word


    @kernel(threads=32)
    def converting(words: Vector[u32], out: Matrix[u32]) -> None:
        item = thread_index()
        lanes: u8x4 = words[item]
        out[item, 0] = lanes
        out[item, 1] = u8x4(words[item])
        out[item, 2] = relabelled(words[item])


    @kernel(threads=32)
    def copying(words: Vector[u32], out: Matrix[u32]) -> None:
        item = thread_index()
        out[item, 0] = words[item]
        out[item, 1] = words[item]
        out[item, 2] = words[item]


    @kernel
    def mismatched(words: Vector[u32], out: Vector[u32]) -> None:
        for item in items(words.size):
            out[item] = bits.absdiff(words[item], u8x4(words[item]))


    @kernel
    def added(words: Vector[u32], out: Vector[u32]) -> None:
        for item in items(words.size):
            out[item] = u8x4(words[item]) + 1


    @kernel
    def narrowed(words: Vector[u32], out: Vector[u32]) -> None:
        for item in items(words.size):
            lanes: u8x4 = words[item]
            low = u8(lanes)
            out[item] = low


    @kernel
    def keyed(words: Vector[u32], out: Vector[u32]) -> None:
        for item in items(words.size):
            out[item] = bits.absdiff(b=words[item], a=7)


    @kernel
    def short(words: Vector[u32], out: Vector[u32]) -> None:
        for item in items(words.size):
            out[item] = bits.absdiff(words[item])
    """
)


@gpu
def test_lanes_convert_to_and_from_a_word_for_free() -> None:
    """A word converts to lanes and back for free and keeps every bit.

    A declaration, a cast, a return and a store convert, adding no instruction.
    """
    words = cp.asarray(np.random.default_rng(5).integers(0, 2**32, 32, dtype=np.uint32))
    outs = [cp.zeros((32, 3), np.uint32) for _ in range(2)]
    ptx = [
        ptx_of(launched, words, out, count=32)
        for launched, out in zip((_LANES.converting, _LANES.copying), outs, strict=True)
    ]

    assert outs[0].get().tolist() == [[word] * 3 for word in words.get().tolist()]
    assert instructions(ptx[0]) == instructions(ptx[1])


@gpu
def test_lanes_take_no_arithmetic_and_no_operation_they_do_not_fit() -> None:
    """A lane operation refuses operands it does not fit, and lanes take no arithmetic.

    The refusal names what the operation takes; an add would carry from lane to lane.
    """
    words, out = cp.arange(4, dtype=np.uint32), cp.zeros(4, np.uint32)
    with pytest.raises(NumbaError, match=r"absdiff takes \(u8x4, u8x4\);.* not \(u32, u8x4\)"):
        _LANES.mismatched[4](words, out)
    with pytest.raises(NumbaError, match="u8x4"):
        _LANES.added[4](words, out)
    with pytest.raises(NumbaError, match=r"absdiff: missing a required argument: 'b'"):
        _LANES.short[4](words, out)
    _LANES.keyed[4](words, out)
    assert out.get().tolist() == [7, 6, 5, 4]


@gpu
def test_lanes_convert_to_no_integer_narrower_than_their_word() -> None:
    """A lanes value read as a `u8` would drop three lanes, so the source goes through `u32`."""
    words, out = cp.arange(4, dtype=np.uint32), cp.zeros(4, np.uint32)
    with pytest.raises(NumbaError, match=r"u8x4 to uint8: lanes are a 32-bit word"):
        _LANES.narrowed[4](words, out)


_SHAPES = executed(
    """
    @kernel(strided=True, threads=32)
    def nested(visits: Vector[i32], count: i32) -> None:
        for outer in items(count):
            for inner in items(count):
                visits[outer] += 1


    @kernel(strided=True, threads=32)
    def branched(visits: Vector[i32], count: i32, flag: i32) -> None:
        if flag > 0:
            for item in items(count):
                visits[item] += 1


    @kernel(strided=True, threads=32)
    def shifted(visits: Vector[i32], count: i32) -> None:
        for item in items(count):
            item = item + 100
            visits[item - 100] = item


    @kernel(strided=True, threads=32)
    def returning(visits: Vector[i32], count: i32) -> None:
        for item in items(count):
            if item % 32 == 5:
                return
            visits[item] = 1


    class Counted(Struct):
        visits: Vector[i32]

        @kernel(strided=True, threads=32)
        def count(self, count: i32) -> None:
            for item in items(count):
                self.visits[item] += 1
    """
)

# One block of 32 threads strides over 70 items, so thread `t` has the items `t`, `t + 32`, ... .
_STRIDED = {
    "nested": lambda i: len(range(i % 32, 70, 32)) if i < 70 else 0,
    "branched": lambda i: int(i < 70),
    "shifted": lambda i: i + 100 if i < 70 else 0,
    "returning": lambda i: int(i < 70 and i % 32 != 5),
}


@gpu
@pytest.mark.parametrize("shape", _STRIDED)
def test_an_items_loop_nests_branches_may_assign_its_item_and_ends_with_a_return(
    *, shape: str
) -> None:
    """Each loop belongs to the thread's own items, which a nested one visits again.

    A branch holds a loop whole, the body may assign `item` without moving the loop, and a `return`
    ends all the items of its thread, so thread 5 visits none.
    """
    visits = cp.zeros(128, np.int32)
    getattr(_SHAPES, shape)[32](visits, 70, *([1] if shape == "branched" else []))

    assert visits.get().tolist() == [_STRIDED[shape](i) for i in range(128)]


@gpu
def test_an_items_loop_of_a_record_kernel_visits_the_items_the_record_is_given() -> None:
    """A kernel defined in a record loops over items as any kernel does."""
    counted = _SHAPES.Counted(cp.zeros(100, np.int32))
    counted.count[32](70)

    assert counted.visits.get().tolist() == [int(i < 70) for i in range(100)]


@gpu
@given(at=st.integers(-(2**40), 2**40))
@settings(deadline=None, max_examples=15)
def test_a_named_value_takes_its_defaults_and_its_keywords_in_the_order_of_its_fields(
    *, at: int
) -> None:
    """Each field converts to its declaration whichever way it is given, a default included."""
    built = executed(
        """
        @device
        def reach(cell: Stepped) -> i64:
            return cell.offset + cell.step


        @kernel
        def build(out: Vector[i64], at: i64) -> None:
            out[0] = reach(Stepped(at))
            out[1] = reach(Stepped(step=at, offset=at + 1))
            out[2] = reach(Stepped(at, step=2))
        """
    )
    out = cp.zeros(3, np.int64)
    built.build[1](out, at)

    offset, step = wrapped(np.uint32, at), wrapped(np.int32, at)
    wanted = [offset + 1, wrapped(np.uint32, at + 1) + step, offset + 2]
    assert out.get().tolist() == [wrapped(np.uint32, value) for value in wanted]


@gpu
def test_a_function_without_parameters_inlines_into_its_callers_unless_it_stays_out_of_line() -> (
    None
):
    """The IR of a caller holds a parameterless function's expression, and a call of the other."""
    inlined = executed(
        """
        @device
        def folded() -> i32:
            return cuda.threadIdx.x


        @device(inline=False)
        def kept() -> i32:
            return cuda.threadIdx.x


        @kernel
        def f(out: Vector[i32]) -> None:
            out[cuda.grid(1)] = folded() + kept()
        """
    ).f
    out = cp.zeros(32, np.int32)
    inlined[32](out)
    defines = re.findall(
        r"define [^@]*@\S*?(folded|kept)",
        inlined.dispatcher.inspect_llvm(next(iter(inlined.dispatcher.overloads))),
    )

    assert defines == ["kept"]
    assert out.get().tolist() == [2 * i for i in range(32)]
