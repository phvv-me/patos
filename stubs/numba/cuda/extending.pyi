"""numba-cuda's `intrinsic`, as device code calls what it decorates.

numba's own stub keeps the typing context in the decorated function's parameters, so a call from
device code reads as missing an argument. The rest of numba-cuda's `extending` is not typed here.
"""

from collections.abc import Callable
from typing import Any, Concatenate

# The first parameter is the typing context numba passes, and the typer returns the signature the
# call compiles to only when it compiles, so the value a call yields has no static type: `Any`.
def intrinsic[C, **P](func: Callable[Concatenate[C, P], object], /) -> Callable[P, Any]: ...
