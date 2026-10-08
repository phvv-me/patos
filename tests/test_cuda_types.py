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
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from types import FunctionType, SimpleNamespace

import cupy as cp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numba import types
from numba.core.errors import TypingError
from numba.core.target_extension import target_override
from numba.core.typing import templates
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda.target import CUDATypingContext
from numba.np.numpy_support import as_dtype, from_dtype

from patos.cuda.primitives import Bitmap, Filter, PairTable
from patos.cuda.runtime import Workspace
from patos.cuda.typed import (
    AnnotationError,
    Constant,
    Kernel,
    Per,
    Struct,
    cuda,
    device,
    i16,
    i32,
    i64,
    kernel,
    number,
    ptx,
    u8,
    u16,
    u32,
    u64,
    unsigned,
)
from patos.cuda.typed.arguments import Argument, argument
from patos.cuda.typed.arithmetic import meet, met, operated
from patos.cuda.typed.declarations import Evaluated, IntLiteral, Kind, Signature
from patos.cuda.typed.scalars import ArrayOf

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
_NUMBA: dict[type, types.Type] = {kind: from_dtype(np.dtype(kind)) for kind in _NAMES}
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

    slots: u64[int]
    mask: u32
    shift: i32


class Lookup(Struct):
    """A record of a record, keys of any unsigned element, the rows a probe fills, and a flag."""

    tables: Tables
    keys: unsigned[int]
    found: u64[int, int]
    exact: bool = True


class Window(Struct):
    """A record whose width is part of its device type, sizing a local array as a literal."""

    values: u64[int]
    width: Constant[int]

    @kernel
    def sums(self, out: u64[int]) -> None:
        """Each thread sums the `width` values from its own, staged through a local array."""
        start = cuda.grid(1)
        if start + self.width <= self.values.size:
            window = cuda.local.array(self.width, np.uint64)
            for offset in range(self.width):
                window[offset] = self.values[start + offset]
            for offset in range(self.width):
                out[start] += window[offset]


@kernel
def total(values: unsigned[int], sums: u64[int]) -> None:
    """Add every value into `sums[0]`."""
    item = cuda.grid(1)
    if item < values.size:
        cuda.atomic.add(sums, 0, u64(values[item]))


@kernel
def looked_up(
    table: PairTable, members: Filter, flags: Bitmap, keys: u64[int], found: i64[int, int]
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


def member(self, out: u64[int]) -> None:
    """A kernel body taking `self`, which only a record class gives it."""


_NAMESPACE: dict[str, Callable | type] = {
    "u8": u8, "u16": u16, "u32": u32, "u64": u64, "i16": i16, "i32": i32, "i64": i64,
    "number": number, "unsigned": unsigned, "Struct": Struct, "Tables": Tables, "Per": Per,
    "cuda": cuda, "device": device, "kernel": kernel, "ptx": ptx,
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

        slots: u64[int]
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
        keys: unsigned[int]
        found: u64[int, int]
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
        row: i64[int], table: i16[int, int], small: u8, signed: i16, wide: u32, flag: bool
    ) -> None:
        """Write each scalar where the host reads it back, `signed` at the table's last corner."""
        row[0] = small
        row[1] = signed
        row[2] = wide
        row[3] = flag
        table[table.shape[0] - 1, table.shape[1] - 1] = signed


    @device
    def first(values: u8[int]) -> u8:
        return values[0]


    @kernel
    def delegate(values: i32[int], out: u8[int]) -> None:
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
        "def f(x: {kind}) -> {kind}[int]:\n    return x\n",
        "a device function returns no array",
    ),
    "record of an array returned": (
        device,
        "def f(x: {kind}) -> Tables:\n    return x\n",
        "a device function returns no array",
    ),
    "array local": (
        device,
        "def f(x: {kind}) -> None:\n    y: {kind}[int] = x\n",
        "`y` declares an array",
    ),
    "subscript declared": (
        device,
        "def f(x: {kind}[int]) -> None:\n    x[0]: {kind} = 1\n",
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


@given(name=st.sampled_from(list(_ELEMENTS.values())), dims=st.integers(1, 3), data=st.data())
def test_a_numeric_type_is_a_scalar_and_by_int_per_dimension_an_array(
    *, name: str, dims: int, data: st.DataObject
) -> None:
    """Called, a numeric type is its numpy scalar; subscripted, an array, of `int`s alone."""
    element = next(kind for kind, short in _ELEMENTS.items() if short == name)
    shape = ", ".join(["int"] * dims)
    wrong = data.draw(st.sampled_from(["3", ":", "float", f"{shape}, 2", f"str, {shape}"]))

    assert annotation(f"{name}[{shape}]") == ArrayOf(element, dims)
    with pytest.raises(TypeError, match=f"{name}\\[.*\\]: an array names each dimension `int`"):
        annotation(f"{name}[{wrong}]")
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
        "def f(out: {kind}[int], x: {other}) -> None:\n    out[0] = {kind}(x)\n",
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
        def f(a: u32, b: u32, out: u64[int]) -> u64:
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

def f[T](x: {first}, pair: Pair, other: T, values: {element}[{shape}]) -> {first}:
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
    shape = ", ".join(["int"] * dims)
    function = dispatched(
        defined(
            _DECLARING,
            first=first,
            second=second,
            third=third,
            element=_ELEMENTS[element],
            shape=shape,
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
        ("def f(value: i32[int]) -> i32:\n    ...\n", AnnotationError, "declares no scalar"),
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
        "Lookup.keys declares unsigned[int], which has no one dtype to take",
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
        "scatter's `row` is array(int32, 1d, C), where i64[int] is declared",
    ),
    "dimensions": (
        lambda: scattered(row=cp.zeros((2, 2), np.int64)),
        TypeError,
        "scatter's `row` is array(int64, 2d, C), where i64[int] is declared",
    ),
    "strided": (
        lambda: scattered(row=cp.zeros(8, np.int64)[::2]),
        TypeError,
        "scatter's `row` is strided, not contiguous, where i64[int] is declared",
    ),
    "transposed": (
        lambda: scattered(table=cp.zeros((3, 2), np.int16).T),
        TypeError,
        "scatter's `table` is strided, not contiguous, where i16[int, int] is declared",
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
        "first's `values` receives array(int32, 1d, C) where u8[int] is declared",
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
    """`unsigned[int]` takes a CuPy or any other device array of each unsigned dtype it meets."""
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
    table, members = PairTable.build(list(entries.items())), Filter.build(entries)
    looked_up[len(keys)](table, members, Bitmap.pack(flags), cp.asarray(keys), found)

    rows = found.get()
    assert rows[:, 0].tolist() == [entries.get(key, -1) for key in keys.tolist()]
    assert rows[: len(entries), 1].all()
    assert rows[:, 2].tolist() == [int(flags[key & 1023]) for key in keys.tolist()]
