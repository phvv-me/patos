"""Device code in plain, annotated Python on numba-cuda, read the way C reads declarations.

Numba ignores annotations, so `device` and `kernel` read them first. Every parameter and every
return carries one. A numeric type is a scalar (`i32`) and, subscripted by its shape, an array of
it (`u8[int]`, `i16[int, int]`, `scalars`). A device function compiles at the types its parameters
declare, a kernel's launch converts its scalars and refuses an array of another element or
dimension count, the return annotation converts every `return`, and a declared local
(`cursor: u64 = 0`) keeps its type through every later assignment (`rewrite`). Integer arithmetic
meets the way C's does (`arithmetic`). An alias of a numeric type (`Index = i32`) declares like
the type it names.

Since the annotations already say what every value converts to, the decorators raise
`AnnotationError` at import for an incomplete signature, a name declared two ways, or a cast that
repeats what an annotation or an operator already does (`checks`). A device function can also be
one PTX block behind an annotated stub (`intrinsics`).

A kernel is an object launched as `kernel[items](*arguments)`, its grid following from what each
item runs on (`kernels`). A `Struct` is its own device type (`records`): device functions and
kernels defined in it with an unannotated `self` are its methods, its operators and its
attributes.
"""

from numba import cuda

from .decorators import device
from .intrinsics import ptx
from .kernels import Kernel, Per, kernel
from .reading import AnnotationError
from .scalars import Constant, i16, i32, i64, number, u8, u16, u32, u64, unsigned
from .struct import Struct

__all__ = [
    "AnnotationError", "Constant", "Kernel", "Per", "Struct", "cuda", "device", "i16", "i32",
    "i64", "kernel", "number", "ptx", "u8", "u16", "u32", "u64", "unsigned",
]  # fmt: skip
