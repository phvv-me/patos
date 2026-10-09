"""Small patches of numba-cuda that patos needs, each naming the change upstream that ends it."""

from functools import cache

from numba.cuda.cudadrv import nvvm


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


def _unverified(unit: nvvm.CompilationUnit) -> None:
    """Leave the module to `compile`, which refuses it as the verifier would."""
