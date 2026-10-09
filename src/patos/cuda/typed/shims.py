"""Small patches of numba-cuda that patos needs, each naming the change upstream that ends it."""

from functools import cache

from numba.cuda.codegen import NRT_LIBRARY, CUDACodeLibrary, JITCUDACodegen
from numba.cuda.cudadrv import nvvm


class _Library(CUDACodeLibrary):
    """A code library that pickles once linked.

    numba-cuda refuses to pickle a library with linking files, a cuco unit's, though the cubin it
    keeps is already the linked result. Upstream: `_reduce_states` should drop the files whenever
    the cubin cache holds one, after which this is dropped.
    """

    def _reduce_states(self) -> dict:
        if not self._cubin_cache:
            return super()._reduce_states()
        files = self._linking_files
        self._linking_files = files & {NRT_LIBRARY}
        try:
            return super()._reduce_states()
        finally:
            self._linking_files = files


@cache
def apply() -> None:
    """Patch numba-cuda, once.

    numba-cuda runs NVVM's verifier over every module before it compiles it (`nvvmVerifyProgram`,
    a full pass over the IR), and `nvvmCompileProgram` then checks the same module and fails on
    the same errors, so the first pass only costs: 1.0 s of the 9.5 s a cold first encode of
    cutok takes. Upstream: `numba.cuda.cudadrv.nvvm.compile_ir` needs a `verify=False` option, or
    an environment switch, after which this is dropped.
    """
    nvvm.CompilationUnit.verify = _unverified
    JITCUDACodegen._library_class = _Library


def _unverified(unit: nvvm.CompilationUnit) -> None:
    """Leave the module to `compile`, which refuses it as the verifier would."""
