"""numba-cuda's `intrinsic`, as device code calls what it decorates.

numba's own stub keeps the typing context in the decorated function's parameters, so a call from
device code reads as missing an argument. The rest of numba-cuda's `extending` is not typed here.
"""

from collections.abc import Callable
from typing import Any, Concatenate

from llvmlite.ir import Value
from numba.core.typing import Signature

# What the typer answers: the signature a call compiles to with the code that lowers it, or `None`
# for the argument types it refuses.
type _Typed = tuple[Signature, Callable[..., Value | None]] | None

# The first parameter is the typing context numba passes, and the typer picks the type a call
# yields from the types of its arguments as it compiles, so the value has no static type: `Any`.
def intrinsic[C, **P](func: Callable[Concatenate[C, P], _Typed], /) -> Callable[P, Any]: ...
