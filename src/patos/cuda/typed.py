"""Device code in plain, annotated Python on numba-cuda, read the way C reads declarations.

Numba ignores annotations, so `device` and `kernel` read them first. Every parameter and every
return carries one. A numeric type is a scalar (`i32`), `Vector[T]` and `Matrix[T]` are arrays of
one and two dimensions of it (`Vector[u8]`, `Matrix[i16]`, `patos.cuda.scalars`), and packed lanes
(`u8x4`) are a 32-bit register read lane by lane. A device function compiles at the types its
parameters declare, a kernel's launch converts its scalars and refuses an array of another element
or dimension count, the return annotation converts every `return`, and a declared local
(`cursor: u64 = 0`) keeps its type through every later assignment. Integer arithmetic meets the
way C's does. An alias of a numeric type (`Index = i32`) declares like the type it names.

Since the annotations already say what every value converts to, the decorators raise
`AnnotationError` at import for an incomplete signature, a name declared two ways, or a cast that
repeats what an annotation or an operator already does. A device function can also be one PTX
block behind an annotated stub (`ptx`), C++ from a header library behind one (`Cxx`), or one name
over several typed implementations, picked by its operands (`dispatched`, `selected`).

A `typing.NamedTuple` of declared fields is a named value, built on the device by calling its
class and held by device functions only, whose device functions with an unannotated `self` are its
methods and attributes. A device function reads who it runs as from `lane()`, `warp_index()` and
the rest of the identity helpers, taking no such parameter.

A kernel is an object launched as `kernel[items](*arguments, stream=stream)` on the stream the
launch names, its grid following from what each item runs on, and loops over its own items as
`for segment in items(count)`. A `Struct` is its own
device type: device functions and kernels defined in it with an unannotated `self` are its methods,
its operators and its attributes, and device code builds one by calling its class.

This module is the surface. `patos.cuda.front` reads, checks and rewrites what is declared without
Numba; `patos.cuda.backend` is the one place numba-cuda's internals are used, and
`patos.cuda.artifacts` keeps each compiled kernel between processes.
"""

from numba import cuda

from .backend.cxx import Compiler, Cxx, Headers, Pinned, cccl, cuco, tree_digest
from .backend.decorators import device
from .backend.identity import (
    block_count,
    block_index,
    lane,
    thread_count,
    thread_in_block,
    thread_index,
    threads_per_block,
    warp_count,
    warp_in_block,
    warp_index,
)
from .backend.intrinsics import ptx
from .backend.kernels import Kernel, Per, kernel
from .backend.overloads import dispatched, selected
from .backend.struct import Struct
from .front.items import items, items_through
from .front.reading import AnnotationError
from .scalars import (
    Constant,
    Matrix,
    Vector,
    f32,
    f64,
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

__all__ = [
    "AnnotationError", "Compiler", "Constant", "Cxx", "Headers", "Kernel", "Matrix", "Per",
    "Pinned", "Struct", "Vector", "block_count", "block_index", "cccl", "cuco", "cuda", "device",
    "dispatched", "f32", "f64", "i8x4", "i16", "i16x2", "i32", "i64", "items", "items_through",
    "kernel", "lane", "number", "ptx", "selected", "thread_count", "thread_in_block",
    "thread_index", "threads_per_block", "tree_digest", "u8", "u8x4", "u16", "u16x2", "u32", "u64",
    "unsigned", "warp_count", "warp_in_block", "warp_index",
]  # fmt: skip
