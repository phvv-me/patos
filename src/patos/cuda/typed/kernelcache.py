"""Kernels compiled once for the sources they are made of and the host they run on.

The first process to compile a kernel pays for it (cold, its folder empty or new, `kernelfolder`)
and each later one loads it (warm). A kernel that cannot be kept compiles in every process: one
made in a function, one that closes over values, one whose source is no file, one that refers to
a global device array, and one whose record class was made in a function.
"""

from contextlib import suppress
from types import FunctionType

from numba import cuda
from numba.cuda.dispatcher import CUDADispatcher

from .kernelfiles import Cache


class Persistent(CUDADispatcher):
    """A kernel dispatcher that keeps its compiled kernels in the cache of the sources it reaches.

    The folder is chosen at the first compile, when every module the kernel names is imported.
    """

    def __init__(self, function: FunctionType) -> None:
        super().__init__(function, targetoptions=cuda.jit(function).targetoptions)
        self.chosen = False

    def compile(self, sig):
        if not self.chosen:
            self.chosen = True
            self.enable_caching()
        return super().compile(sig)

    def enable_caching(self) -> None:
        """Keep the kernels in the cache, unless they are made in a function or have no file."""
        if not self.py_func.__closure__:
            with suppress(RuntimeError):
                self._cache = Cache(self.py_func)

    def forget(self) -> None:
        """Delete what was kept of this kernel, which a launch found it cannot run as built.

        A kernel loaded later would skip the checks of the compile that built it.
        """
        self._cache.flush()
