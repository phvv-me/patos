"""`patos.cuda.typed` on a host without a GPU.

numba-cuda decorates lazily, so the annotation checks run at decoration, the rewritten functions
execute as plain Python through `py_func`, and the typing rules resolve in a CUDA typing context
without compiling for a device.
"""

import ast
import importlib
import importlib.util
import itertools
import linecache
import operator
import re
import textwrap
from collections.abc import Callable, Sequence
from typing import NamedTuple

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# The `cuda` extra carries numba-cuda, so without it there is nothing here to test.
if importlib.util.find_spec("numba") is None:
    pytest.skip("patos.cuda needs the `cuda` extra (numba-cuda)", allow_module_level=True)

import numpy as np
from numba import types
from numba.core.errors import TypingError
from numba.core.target_extension import target_override
from numba.core.typing import templates
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda.target import CUDATypingContext
from numba.np.numpy_support import from_dtype

from patos.cuda.typed import (
    AnnotationError,
    Array,
    Struct,
    device,
    i16,
    i32,
    i64,
    kernel,
    ptx,
    u8,
    u16,
    u32,
    u64,
)
from patos.cuda.typed.arithmetic import _ARITHMETIC, _COMPARISONS, _SHIFTS, meet, met, operated
from patos.cuda.typed.declarations import Kind, Literal, Signature

type Scalar = type[np.integer]
type Reading = Scalar | Literal

_NAMES: dict[Scalar, str] = {
    i16: "i16", i32: "i32", i64: "i64", u8: "u8", u16: "u16", u32: "u32", u64: "u64",
}  # fmt: skip
_NUMBA: dict[type, types.Type] = {kind: from_dtype(np.dtype(kind)) for kind in _NAMES}
_NUMBA[bool] = types.boolean
scalars = st.sampled_from(list(_NAMES))
# numpy types a u64 met with a signed integer as a float, which no conversion round-trips.
not_u64 = st.sampled_from([kind for kind in _NAMES if kind is not u64])
bounded = st.integers(-(2**40), 2**40)
# The edges of every scalar type, where a literal fits or does not, inside what Numba can type.
_EDGES = [
    edge
    for bit in (7, 8, 15, 16, 31, 32, 63, 64)
    for edge in (-(2**bit), 2**bit - 1, 2**bit)
    if -(2**63) <= edge < 2**64
]
literals = st.one_of(st.integers(-(2**63), 2**64 - 1), st.sampled_from(_EDGES)).map(Literal)
operands = st.one_of(scalars, literals)
# A literal past int64 types as uint64 whatever it meets, which the shift reading leaves out.
int64_operands = st.one_of(scalars, literals.filter(lambda literal: literal.value < 2**63))
_BINARY: dict[type[ast.operator], Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}  # fmt: skip
_OPERATORS = [op for op in (*_ARITHMETIC, *_COMPARISONS, *_SHIFTS) if op not in (min, max)]
_SOURCES = itertools.count()


class Tables(Struct):
    """A record of one array and two scalars, the second one defaulted."""

    slots: Array[u64]
    mask: u32
    shift: i32 = i32(3)


class PlainTables(NamedTuple):
    """The same fields as a plain named tuple, which the host builds unchecked."""

    slots: Array[u64]
    mask: u32
    shift: i32 = i32(3)


def defined(template: str, **kinds: Scalar) -> Callable:
    """The function `f` that `template` defines, not decorated yet.

    The `{field}`s of the template are filled with the short names of `kinds`. It runs in a
    namespace of the scalars, `Array`, `device`, `kernel`, `ptx` and the two records, and its text
    is registered under a made-up file so that `inspect.getsource` finds it as it does a real
    module's.
    """
    shorts = {field: _NAMES[kind] for field, kind in kinds.items()}
    text = textwrap.dedent(template).lstrip("\n").format(**shorts)
    filename = f"<patos-cuda-test-{next(_SOURCES)}>"
    linecache.cache[filename] = (len(text), None, text.splitlines(keepends=True), filename)
    namespace: dict[str, Callable | type] = {
        **{short: kind for kind, short in _NAMES.items()},
        "Array": Array,
        "device": device,
        "kernel": kernel,
        "ptx": ptx,
        "Tables": Tables,
        "PlainTables": PlainTables,
    }
    exec(compile(text, filename, "exec"), namespace)
    return namespace["f"]


def dispatched(function: Callable, *, decorator: Callable = device) -> CUDADispatcher:
    """`function` as `decorator` compiles it."""
    dispatcher = decorator(function)
    assert isinstance(dispatcher, CUDADispatcher)
    return dispatcher


def recorded(function: Callable) -> Signature:
    """What the decorator recorded of the annotations of the rewritten `function`."""
    return function.__dict__["device_signature"]


def host(template: str, **kinds: Scalar) -> Callable:
    """The rewritten function `template` defines as plain Python, annotations converting."""
    return dispatched(defined(template, **kinds)).py_func


def rejection(function: Callable, *, decorator: Callable = device) -> str:
    """The `AnnotationError` message `decorator` raises for `function`."""
    with pytest.raises(AnnotationError) as caught:
        decorator(function)
    return str(caught.value)


def verdict(function: Callable) -> str:
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


def numba_type(kind: Kind) -> types.Type:
    """The Numba type a reading of `kind` stands for."""
    assert kind is not None
    return types.IntegerLiteral(kind.value) if isinstance(kind, Literal) else _NUMBA[kind]


def is_negative_literal(reading: Reading) -> bool:
    """Whether `reading` is a literal below zero."""
    return isinstance(reading, Literal) and reading.value < 0


def decided(*, left: Reading, right: Reading) -> types.Type | None:
    """What `met` decides in the order Numba asks a rule that prefers literals.

    That order is the operands' literals first, then their plain types.
    """
    literal = met(numba_type(left), numba_type(right))
    return (
        literal
        if literal is not None
        else met(types.unliteral(numba_type(left)), types.unliteral(numba_type(right)))
    )


@pytest.fixture(scope="module")
def context() -> CUDATypingContext:
    """A CUDA typing context carrying the rules the module registers."""
    typing = CUDATypingContext()
    typing.refresh()
    return typing


def resolved(
    context: CUDATypingContext, function: Callable, *arguments: types.Type
) -> templates.Signature:
    """The signature `function` takes on `arguments` on the CUDA target."""
    with target_override("cuda"):
        signature = context.resolve_function_type(function, arguments, {})
    assert signature is not None
    return signature


@given(kind=scalars)
def test_an_incomplete_signature_names_the_parameter_and_the_return(*, kind: Scalar) -> None:
    """A parameter without an annotation and a missing return annotation are both named."""
    parameter = rejection(defined("def f(x) -> {kind}:\n    return x\n", kind=kind))
    result = rejection(defined("def f(x: {kind}):\n    return x\n", kind=kind))

    assert "parameter `x` has no annotation" in parameter
    assert "the return has no annotation" in result


def test_every_problem_is_reported_on_a_line_of_its_own() -> None:
    """The message lists each issue as `file:line: function: message`, not only the first."""
    lines = rejection(defined("def f(x, y):\n    pass\n")).splitlines()

    assert len(lines) == 3
    assert all(re.match(r"<patos-cuda-test-\d+>:\d+: f: ", line) for line in lines)


@given(kind=scalars)
def test_a_kernel_returns_none(*, kind: Scalar) -> None:
    """A kernel annotated with anything but None is rejected, one annotated None is not."""
    returns_value = defined("def f(x: {kind}) -> {kind}:\n    return x\n", kind=kind)
    returns_none = defined("def f(x: {kind}) -> None:\n    pass\n", kind=kind)

    function = dispatched(returns_none, decorator=kernel).py_func

    assert "a kernel returns None" in rejection(returns_value, decorator=kernel)
    assert recorded(function).returns is type(None)


@given(kind=scalars)
def test_a_return_agrees_with_the_return_annotation(*, kind: Scalar) -> None:
    """A return carries a value exactly where the annotation names one, element by element."""
    value = rejection(defined("def f(x: {kind}) -> None:\n    return x\n", kind=kind))
    nothing = rejection(defined("def f(x: {kind}) -> {kind}:\n    return\n", kind=kind))
    whole = rejection(
        defined("def f(x: {kind}) -> tuple[{kind}, {kind}]:\n    return x\n", kind=kind)
    )

    assert "returns a value where None is declared" in value
    assert f"returns nothing where {_NAMES[kind]} is declared" in nothing
    assert "return the 2 elements so each converts" in whole


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("def f(x: str) -> None:\n    pass\n", "`str` names no device type"),
        ("def f(x: i32) -> Array[i32]:\n    return x\n", "a device function returns no array"),
        ("def f(x: i32) -> None:\n    y: Array[i32] = x\n", "declares an array"),
        ("def f(x: Array[i32]) -> None:\n    x[0]: i32 = 1\n", "is no name to declare"),
    ],
)
def test_annotations_must_name_device_types(*, template: str, message: str) -> None:
    """An annotation outside the device types, or where it cannot be carried, is rejected."""
    assert message in rejection(defined(template))


@given(kind=scalars, other=scalars)
def test_a_name_is_declared_once(*, kind: Scalar, other: Scalar) -> None:
    """A second declaration of a local or of a parameter is rejected, even when it agrees."""
    local = defined(
        "def f(x: {kind}) -> None:\n    y: {kind} = 0\n    y: {other} = 1\n",
        kind=kind,
        other=other,
    )
    parameter = defined("def f(x: {kind}) -> None:\n    x: {other} = 1\n", kind=kind, other=other)

    assert f"`y` is declared {_NAMES[kind]} already" in rejection(local)
    assert "parameter `x` is declared again" in rejection(parameter)


@given(kind=scalars)
def test_a_loop_variable_is_not_declared(*, kind: Scalar) -> None:
    """The iterable types a loop variable, so declaring it as well is rejected."""
    loop = defined(
        """
        def f(n: i32) -> None:
            i: {kind} = 0
            for i in range(n):
                pass
        """,
        kind=kind,
    )

    assert "loop variable `i` is declared" in rejection(loop)


_REDUNDANT = {
    "value already of the type": (
        "def f(x: {kind}) -> None:\n    y = {kind}(x)\n",
        "`x` is already {kind}",
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
        "def f(out: Array[{kind}], x: {other}) -> None:\n    out[0] = {kind}(x)\n",
        "the store into `out` converts to {kind}",
    ),
}


@pytest.mark.parametrize(("template", "reason"), _REDUNDANT.values(), ids=_REDUNDANT)
@given(kind=scalars, other=scalars)
def test_a_cast_an_annotation_repeats_is_redundant(
    *, template: str, reason: str, kind: Scalar, other: Scalar
) -> None:
    """A cast to the type an annotation already converts to is rejected, naming that annotation."""
    message = rejection(defined(template, kind=kind, other=other))

    assert "is a redundant cast" in message
    assert reason.format(kind=_NAMES[kind]) in message


@given(kind=scalars, other=scalars)
def test_a_callees_parameter_converts_its_argument(*, kind: Scalar, other: Scalar) -> None:
    """A cast of an argument to the type the callee's annotated parameter has is redundant."""
    caller = defined(
        """
        @device
        def callee(x: {kind}) -> {kind}:
            return x

        def f(y: {other}) -> None:
            z = callee({kind}(y))
        """,
        kind=kind,
        other=other,
    )

    assert f"`callee`'s parameter converts to {_NAMES[kind]}" in rejection(caller)


_STATEMENTS = {
    "right operand": "c = a + {kind}(b)",
    "left operand": "c = {kind}(b) * a",
    "comparison": "c = a < {kind}(b)",
    "min": "c = min(a, {kind}(b))",
    "augmented": "a += {kind}(b)",
}


def casting(statement: str, *, kind: Scalar, other: Scalar) -> Callable:
    """`a` of type `kind` meeting `b` of type `other` in `statement`, which casts `b` to `kind`."""
    template = f"def f(a: {{kind}}, b: {{other}}) -> None:\n    {statement}\n"
    return defined(template, kind=kind, other=other)


@pytest.mark.parametrize("statement", _STATEMENTS.values(), ids=_STATEMENTS)
@pytest.mark.parametrize(
    ("kind", "other"),
    [(u64, i32), (u64, u8), (u64, u64), (i64, i16), (i64, u32), (u32, i16), (u32, u8)],
)
def test_a_cast_the_operator_repeats_is_redundant(
    *, statement: str, kind: Scalar, other: Scalar
) -> None:
    """Where the operator already meets `b` in the type of `a`, casting `b` to it is rejected."""
    message = rejection(casting(statement, kind=kind, other=other))

    assert f"`a` is {_NAMES[kind]}, so the operator converts it" in message


@pytest.mark.parametrize("statement", _STATEMENTS.values(), ids=_STATEMENTS)
@pytest.mark.parametrize(
    ("kind", "other"), [(i32, i16), (i32, u32), (u32, u64), (u32, i64), (i64, u64), (u8, i16)]
)
def test_a_cast_the_operator_would_not_make_is_kept(
    *, statement: str, kind: Scalar, other: Scalar
) -> None:
    """Where the operator meets the operands elsewhere, the cast changes the result and stays."""
    device(casting(statement, kind=kind, other=other))


@given(kind=scalars, other=scalars)
@settings(deadline=None)
def test_a_cast_is_rejected_only_where_numba_types_the_sum_as_the_cast(
    context: CUDATypingContext, *, kind: Scalar, other: Scalar
) -> None:
    """Soundness: a cast the decorator calls redundant never changes what Numba types."""
    message = verdict(casting("c = a + {kind}(b)", kind=kind, other=other))
    if message:
        assume("so the operator converts it" in message)
        signature = resolved(context, operator.add, numba_type(kind), numba_type(other))
        assert signature.return_type == numba_type(kind)


@given(kind=scalars, other=scalars)
def test_a_cast_to_another_type_is_not_redundant(*, kind: Scalar, other: Scalar) -> None:
    """A conversion nothing else performs decorates fine and records the signature."""
    assume(kind is not other)
    template = "def f(x: {other}) -> None:\n    y = {kind}(x)\n"
    function = dispatched(defined(template, kind=kind, other=other)).py_func

    assert recorded(function).parameters == (other,)


def test_a_clean_function_decorates_and_keeps_its_docstring() -> None:
    """Widening before a product repeats nothing, and the rewrite keeps docstring and name."""
    function = dispatched(
        defined(
            '''
            def f(a: u32, b: u32, out: Array[u64]) -> u64:
                """Widen before the product."""
                total: u64 = u64(a) * b
                out[0] = total
                return total
            '''
        )
    ).py_func

    assert function.__doc__ == "Widen before the product."
    assert function.__qualname__ == "f"
    assert recorded(function).returns is u64


def test_a_rebuilt_function_reads_its_module_globals_as_they_grow() -> None:
    """A device function calling one the module defines further down finds it at compile time."""
    function = defined("def f(x: i32) -> i32:\n    return x\n")

    assert dispatched(function).py_func.__globals__ is function.__globals__


def test_a_type_parameter_is_left_alone_and_a_type_alias_declares_its_type() -> None:
    """A generic parameter converts nothing, and `type Index = i32` converts like `i32`."""
    generic = host("def f[T](x: T) -> T:\n    return x\n")
    aliased = host("type Index = i32\n\ndef f(x: Index) -> Index:\n    return x\n")

    assert generic("unchanged") == "unchanged"
    assert type(aliased(np.int64(7))) is i32


@given(kinds=st.tuples(scalars, scalars, scalars), given=st.tuples(scalars, scalars, scalars))
def test_a_device_function_compiles_at_its_declared_parameter_types(
    *, kinds: Sequence[Scalar], given: Sequence[Scalar]
) -> None:
    """Numba compiles a device function at the parameter types it declares.

    Scalars and tuples, through a type alias too, take their declared types whatever the caller
    passes, and a type parameter keeps the type it is passed.
    """
    template = """
    type Pair = tuple[{second}, {third}]

    def f[T](x: {first}, pair: Pair, other: T) -> {first}:
        return x
    """
    function = dispatched(defined(template, first=kinds[0], second=kinds[1], third=kinds[2]))
    passed = (_NUMBA[given[0]], types.UniTuple(_NUMBA[given[1]], 2), _NUMBA[given[2]])
    compiled: list[tuple[types.Type, ...]] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            CUDADispatcher, "compile_device", lambda _, args, _returns=None: compiled.append(args)
        )
        function.compile_device(passed)

    pair = types.BaseTuple.from_types([_NUMBA[kinds[1]], _NUMBA[kinds[2]]])
    assert compiled == [(_NUMBA[kinds[0]], pair, _NUMBA[given[2]])]


@given(kind=not_u64, a=bounded, b=bounded)
def test_a_declared_local_keeps_its_type_across_assignments(
    *, kind: Scalar, a: int, b: int
) -> None:
    """Plain and augmented assignment convert back to the declaration, wrapping at each step."""
    template = """
    def f(a: i64, b: i64) -> i64:
        cursor: {kind}
        cursor = a
        cursor = cursor + b
        cursor += b
        return cursor
    """
    expected = wrapped(kind, a)
    expected = wrapped(kind, expected + b)
    expected = wrapped(kind, expected + b)

    assert host(template, kind=kind)(np.int64(a), np.int64(b)) == wrapped(i64, expected)


@given(kind=scalars, a=bounded, b=bounded)
def test_unpacking_converts_only_the_declared_targets(*, kind: Scalar, a: int, b: int) -> None:
    """In `low, high = a, b` only the declared `low` converts."""
    template = """
    def f(a: i64, b: i64) -> tuple[i64, i64]:
        low: {kind} = 0
        high = 0
        low, high = a, b
        return low, high
    """
    low, high = host(template, kind=kind)(np.int64(a), np.int64(b))

    assert (low, high) == (wrapped(i64, wrapped(kind, a)), b)


@given(kinds=st.tuples(scalars, scalars, scalars), a=bounded, b=bounded)
def test_a_return_converts_each_element_of_nested_tuples(
    *, kinds: tuple[Scalar, Scalar, Scalar], a: int, b: int
) -> None:
    """Tuple and nested tuple returns convert element by element to the annotated types."""
    template = """
    def f(a: i64, b: i64) -> tuple[{first}, tuple[{second}, {third}]]:
        return a, (b, a)
    """
    function = host(template, first=kinds[0], second=kinds[1], third=kinds[2])
    head, (middle, tail) = function(np.int64(a), np.int64(b))

    assert (type(head), type(middle), type(tail)) == kinds
    assert [head, middle, tail] == held(kinds, [a, b, a])


@given(kind=scalars, flag=st.booleans(), a=bounded, b=bounded)
def test_a_conditional_converts_branch_by_branch(
    *, kind: Scalar, flag: bool, a: int, b: int
) -> None:
    """Whichever branch runs, the return and the declared local hold `kind`."""
    returned = host(
        "def f(flag: bool, a: i64, b: i64) -> {kind}:\n    return a if flag else b\n", kind=kind
    )
    declared = host(
        """
        def f(flag: bool, a: i64, b: i64) -> i64:
            chosen: {kind} = a if flag else b
            return chosen
        """,
        kind=kind,
    )
    arguments = (flag, np.int64(a), np.int64(b))
    expected = wrapped(kind, a if flag else b)

    assert type(returned(*arguments)) is kind
    assert returned(*arguments) == expected
    assert declared(*arguments) == wrapped(i64, expected)


@given(kind=scalars, other=scalars)
@settings(deadline=None)
def test_an_array_parameter_accepts_only_its_element_type(
    context: CUDATypingContext, *, kind: Scalar, other: Scalar
) -> None:
    """The check an `Array[T]` parameter leaves in the body fails typing for any other dtype."""
    function = dispatched(
        defined("def f(a: Array[{kind}]) -> None:\n    pass\n", kind=kind)
    ).py_func
    (name,) = [name for name in function.__code__.co_names if name.startswith("check_")]
    check = getattr(function.__globals__["_patos_expectations"], name)
    arguments = (types.Array(numba_type(other), 1, "C"), types.NumberClass(numba_type(kind)))
    context.refresh()

    if kind is other:
        assert resolved(context, check, *arguments).return_type == types.none
    else:
        message = (
            rf"f's `a` receives array\({numba_type(other)}, 1d, C\) "
            f"where an Array of {numba_type(kind)} is declared"
        )
        with pytest.raises(TypingError, match=message):
            resolved(context, check, *arguments)


@given(left=operands, right=operands)
def test_the_static_meet_agrees_with_the_runtime_rule(*, left: Reading, right: Reading) -> None:
    """Wherever the runtime rule `met` decides, `meet` reads the same type, and only there."""
    runtime = met(numba_type(left), numba_type(right))
    static = meet(left, right)

    if runtime is None:
        assert static is None
    else:
        assert numba_type(static) == runtime


@given(other=st.one_of(operands, st.just(bool)), narrow=st.sampled_from([i16, u8, u16]))
def test_integers_narrower_than_32_bits_meet_as_i32(
    *, other: Reading | type[bool], narrow: Scalar
) -> None:
    """The C integer promotion holds statically and at runtime, on either side of the operator."""
    assert meet(narrow, other) is meet(i32, other)
    assert meet(other, narrow) is meet(other, i32)
    assert met(numba_type(narrow), numba_type(other)) == met(types.int32, numba_type(other))
    assert met(numba_type(other), numba_type(narrow)) == met(numba_type(other), types.int32)


@given(op=st.sampled_from(list(_BINARY)), left=int64_operands, right=int64_operands)
@settings(deadline=None)
def test_the_static_reading_agrees_with_the_registered_typing(
    context: CUDATypingContext, *, op: type[ast.operator], left: Reading, right: Reading
) -> None:
    """Whatever type `operated` reads for `left op right`, Numba's typing gives it too."""
    reading = operated(op(), left, right)
    assume(reading is not None)
    # `u32` meeting a literal is the gap `test_a_literal_takes_the_type_of_the_u32_it_meets` holds.
    assume(not (u32 in (left, right) and Literal in (type(left), type(right))))
    # A negative literal shifted by a u64 count: Numba meets it with the plain int64 in the u64.
    assume(not (op in (ast.LShift, ast.RShift) and is_negative_literal(left) and right is u64))

    signature = resolved(context, _BINARY[op], numba_type(left), numba_type(right))

    assert signature.return_type == numba_type(reading)


@given(op=st.sampled_from(_OPERATORS), left=operands, right=operands)
@settings(deadline=None)
def test_the_registered_rules_type_operators_as_the_runtime_rule_decides(
    context: CUDATypingContext, *, op: Callable, left: Reading, right: Reading
) -> None:
    """Where `met` decides, an operator takes both operands in it and answers it or a bool."""
    meeting = decided(left=left, right=right)
    assume(meeting is not None and (op not in _SHIFTS or meeting in (types.int64, types.uint64)))

    signature = resolved(context, op, numba_type(left), numba_type(right))

    assert signature.args == (meeting, meeting)
    assert signature.return_type == (types.boolean if op in _COMPARISONS else meeting)


def test_a_literal_takes_the_type_of_the_u32_it_meets(context: CUDATypingContext) -> None:
    """Numba asks the rule about the literal before its plain `int64`, so `u32 + 1` stays u32."""
    signature = resolved(context, operator.add, types.uint32, types.IntegerLiteral(1))

    assert meet(u32, Literal(1)) is u32
    assert signature.return_type == types.uint32


@pytest.mark.parametrize("module", ["warp", "hash"])
def test_the_ported_modules_pass_their_own_annotation_checks(*, module: str) -> None:
    """Importing a module decorates its device functions, which raises on a wrong annotation."""
    assert importlib.import_module(f"patos.cuda.primitives.{module}")


@given(mask=st.integers(0, 2**32 - 1), other=st.integers(0, 2**32 - 1))
def test_a_struct_converts_its_scalars_where_it_is_built(*, mask: int, other: int) -> None:
    """Building or replacing a record converts every scalar to its declaration."""
    tables = Tables(np.zeros(2, np.uint64), mask)
    replaced = tables._replace(mask=other)

    assert (type(tables.mask), type(tables.shift), tables.mask) == (u32, i32, mask)
    assert (type(replaced.mask), replaced.mask, replaced.slots) == (u32, other, tables.slots)
    assert tables._fields == ("slots", "mask", "shift")


@given(dtype=st.sampled_from([np.uint8, np.int32, np.int64, np.uint32, np.float64]))
def test_a_struct_refuses_an_array_of_another_element(*, dtype: type[np.generic]) -> None:
    """An array of the wrong element fails where the record is built or replaced."""
    tables = Tables(np.zeros(2, np.uint64), 1)
    wrong = np.zeros(2, dtype)

    with pytest.raises(TypeError, match=f"Tables.slots holds {np.dtype(dtype)}, not the u64"):
        Tables(wrong, 1)
    with pytest.raises(TypeError, match="Tables.slots holds"):
        tables._replace(slots=wrong)


@pytest.mark.parametrize("record", ["Tables", "PlainTables"])
def test_a_record_field_reads_as_its_declared_type(*, record: str) -> None:
    """A field of a record parameter, a `Struct` or a plain named tuple, has its declared type."""
    message = rejection(
        defined(
            f"def f(tables: {record}) -> u64:\n    return u64(u32(tables.mask) + tables.shift)\n"
        )
    )

    assert "`u32(tables.mask)` is a redundant cast: `tables.mask` is already u32" in message


@given(kind=st.sampled_from([i16, i32, i64, u16, u32, u64]), given=scalars)
def test_a_ptx_call_takes_the_stubs_declared_types(
    context: CUDATypingContext, *, kind: Scalar, given: Scalar
) -> None:
    """Numba types a call of a PTX stub at its declared types, casting what the caller passes."""
    stub = ptx("mov.b64 $result, $value;")(
        defined("def f(value: {kind}) -> {kind}:\n    ...\n", kind=kind)
    )
    # An intrinsic registers its typing as it is made, after the module's context was built.
    context.refresh()
    signature = resolved(context, stub, _NUMBA[given])

    assert (signature.args, signature.return_type) == ((_NUMBA[kind],), _NUMBA[kind])


def test_a_ptx_stub_declares_scalars_and_names_only_its_operands() -> None:
    """A PTX stub that cannot be lowered is refused where it is defined.

    That is a stub without scalar types or with a byte operand, or a template naming an operand
    the stub lacks.
    """
    with pytest.raises(AnnotationError, match="16, 32 or 64 bits"):
        ptx("mov.b32 $result, $value;")(defined("def f(value: u8) -> i32:\n    ...\n"))
    with pytest.raises(AnnotationError, match="declares no scalar"):
        ptx("mov.b32 $result, $value;")(defined("def f(value: Array[i32]) -> i32:\n    ...\n"))
    with pytest.raises(KeyError, match="other"):
        ptx("mov.b32 $result, $other;")(defined("def f(value: i32) -> i32:\n    ...\n"))


def test_the_checks_read_a_ptx_stubs_signature() -> None:
    """A cast of a PTX stub's result to the type it already returns is redundant."""
    template = """
    @ptx("mov.b32 $result, $value;")
    def g(value: i32) -> i32:
        ...

    def f(x: i32) -> i64:
        return i32(g(x))
    """

    assert "`i32(g(x))` is a redundant cast: `g(x)` is already i32" in rejection(defined(template))
