"""Device code in plain, annotated Python on numba-cuda, read the way C reads declarations.

Numba ignores annotations, so `device` and `kernel` read them first. Every parameter and every
return carries one. A scalar parameter converts on entry, an `Array[T]` parameter fails
compilation when the array it receives holds anything but `T`, the return annotation converts
every `return`, and a declared local (`cursor: u64 = 0`) keeps its type through every later
assignment (`rewrite`). Integer arithmetic meets the way C's does (`arithmetic`). A `type` alias of
a scalar (`type Index = i32`) declares like the type it names, and a `Struct` bundles a kernel's
many arguments into one named, frozen record checked where the host builds it (`struct`).

Since the annotations already say what every value converts to, the decorators raise
`AnnotationError` at import for an incomplete signature, a name declared two ways, or a cast that
repeats what an annotation or an operator already does (`checks`). A device function can also be
one PTX block behind an annotated stub (`intrinsics`).
"""

from numba import cuda

from .decorators import device, kernel
from .intrinsics import ptx
from .reading import AnnotationError
from .scalars import Array, i16, i32, i64, u8, u16, u32, u64
from .struct import Struct

__all__ = [
    "AnnotationError", "Array", "Struct", "cuda", "device", "i16", "i32", "i64", "kernel", "ptx",
    "u8", "u16", "u32", "u64",
]  # fmt: skip
