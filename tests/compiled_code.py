"""What a launched kernel compiled to, for the tests that read its LLVM IR, PTX or SASS."""

from numba.cuda.codegen import disassemble_cubin
from numba.cuda.np.numpy_support import as_dtype
from patos.cuda.backend.compiler import Built, built
from patos.cuda.typed import Kernel


def compiled(launched: Kernel, element: type | None = None) -> Built:
    """`launched` compiled again for a signature it was launched with.

    element: the dtype of the first array of that signature, when it launched with several.
    """
    kinds = next(
        kinds
        for kinds, _ in launched.compiled.values()
        if element is None or as_dtype(kinds[0].dtype) == element
    )
    return built(launched.rebuilt, kinds)


def llvm(launched: Kernel, element: type | None = None) -> str:
    """The LLVM IR of `launched` with every device function it calls."""
    return compiled(launched, element).library.get_llvm_str()


def ptx(launched: Kernel) -> str:
    return compiled(launched).library.get_asm_str()


def sass(launched: Kernel, element: type | None = None) -> str:
    return disassemble_cubin(compiled(launched, element).cubin.code)
