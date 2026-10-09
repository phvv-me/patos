"""Device code in plain, annotated Python on numba-cuda, read the way C reads declarations.

Numba ignores annotations, so `device` and `kernel` read them first. Every parameter and every
return carries one. A numeric type is a scalar (`i32`), `Vector[T]` and `Matrix[T]` are arrays of
one and two dimensions of it (`Vector[u8]`, `Matrix[i16]`, `patos.cuda.scalars`), and packed lanes
(`u8x4`) are a 32-bit register read lane by lane (`lanes`). A device function compiles at the
types its parameters declare, a kernel's launch converts its scalars and refuses an array of
another element or dimension count, the return annotation converts every `return`, and a declared
local (`cursor: u64 = 0`) keeps its type through every later assignment (`rewrite`). Integer
arithmetic meets the way C's does (`arithmetic`). An alias of a numeric type (`Index = i32`)
declares like the type it names.

Since the annotations already say what every value converts to, the decorators raise
`AnnotationError` at import for an incomplete signature, a name declared two ways, or a cast that
repeats what an annotation or an operator already does (`checks`). A device function can also be
one PTX block behind an annotated stub (`intrinsics`), C++ from a header library behind one
(`cxx`), or one name over several typed implementations, picked by its operands (`overloads`).

A `typing.NamedTuple` of declared fields is a named value, built on the device by calling its class
and held by device functions only (`declarations`), whose device functions with an unannotated
`self` are its methods and attributes (`values`). A device function reads who it runs as from
`lane()`, `warp_index()` and the rest of `identity`, taking no such parameter.

A kernel is an object launched as `kernel[items](*arguments)`, its grid following from what each
item runs on (`kernels`), and loops over its own items as `for segment in items(count)`
(`items`). A `Struct` is its own device type (`records`): device functions and
kernels defined in it with an unannotated `self` are its methods, its operators and its
attributes.
"""

from numba import cuda

from ..scalars import (
    Constant,
    Matrix,
    Vector,
    i8x4,
    i16,
    i16x2,
    i32,
    i64,
    number,
    u8,
    u8x4,
    u16,
    u16x2,
    u32,
    u64,
    unsigned,
)
from .cxx import Compiler, Cxx, Headers, Pinned, cccl, cuco, tree_digest
from .decorators import device
from .identity import (
    block_index,
    lane,
    thread_in_block,
    thread_index,
    warp_in_block,
    warp_index,
)
from .intrinsics import ptx
from .items import items, items_through
from .kernels import Kernel, Per, kernel
from .overloads import dispatched
from .reading import AnnotationError
from .struct import Struct

__all__ = [
    "AnnotationError", "Compiler", "Constant", "Cxx", "Headers", "Kernel", "Matrix", "Per",
    "Pinned", "Struct", "Vector", "block_index", "cccl", "cuco", "cuda", "device", "dispatched",
    "i8x4", "i16", "i16x2", "i32", "i64", "items", "items_through", "kernel", "lane", "number",
    "ptx", "thread_in_block", "thread_index", "tree_digest", "u8", "u8x4", "u16", "u16x2", "u32",
    "u64", "unsigned", "warp_in_block", "warp_index",
]  # fmt: skip
