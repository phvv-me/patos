"""Kernels compiled to a cubin: numba-cuda types and lowers a kernel to LLVM IR, NVVM compiles it
with every device function it calls, and numba-cuda's linker links the result with the units they
name, such as a C++ unit's LTO IR.

This is what numba-cuda's `_Kernel` and `CUDACodeLibrary.get_cubin` do for a kernel, at the same
options and so to the same code, less two passes. NVVM's verifier is not run: numba-cuda verifies
every module before compiling it (`nvvmVerifyProgram`, a full pass over the IR), and
`nvvmCompileProgram` refuses the same IR with the same errors. And under LTO, numba-cuda's default
where nvJitLink is found, no PTX is compiled only to be searched for the runtime functions a
kernel needs linked; the LLVM IR names them as well. Upstream: `nvvm.compile_ir` needs a
`verify=False` option, after which `_nvvm` is that call.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FunctionType

from cuda.core import ObjectCode
from numba import cuda, types
from numba.cuda import dispatcher
from numba.cuda.codegen import CUDACodeLibrary
from numba.cuda.compiler import compile_cuda, kernel_fixup
from numba.cuda.core.compiler_lock import global_compiler_lock
from numba.cuda.cudadrv import driver, nvrtc, nvvm

from ...bases import FrozenModel, Runtime
from ..artifacts import Cubin
from .arithmetic import rules

# The helper library numba-cuda links when a kernel reshapes an array.
_RESHAPING = Path(dispatcher.__file__).with_name("reshape_funcs.cu")


class Built(FrozenModel):
    """A kernel compiled in this process.

    keepable: whether another process can load `cubin`; it cannot when the kernel refers to a
        global device array, whose address the code holds.
    """

    cubin: Cubin
    library: Runtime[CUDACodeLibrary]
    keepable: bool


@global_compiler_lock
def built(function: FunctionType, kinds: Sequence[types.Type]) -> Built:
    """`function` compiled as a kernel taking arguments of `kinds`, at `cuda.jit`'s options."""
    options = cuda.jit(function).targetoptions
    library, entry = _lowered(function, kinds, options)
    held = library.referenced_objects.values()
    return Built(
        cubin=Cubin(code=_linked(library, options), entry=entry),
        library=library,
        keepable=not any(hasattr(each, "__cuda_array_interface__") for each in held),
    )


def _lowered(
    function: FunctionType, kinds: Sequence[types.Type], options: Mapping
) -> tuple[CUDACodeLibrary, str]:
    """The library holding `function`'s LLVM IR as a kernel, and the kernel's name in it."""
    with rules():
        result = compile_cuda(
            function, types.void, kinds, debug=options["debug"], lineinfo=options["lineinfo"],
            forceinline=options["forceinline"], fastmath=options["fastmath"],
            nvvm_options=_nvvm_options(options), cc=cuda.get_current_device().compute_capability,
            max_registers=options.get("max_registers"), lto=options["lto"],
        )  # fmt: skip
    library, entry = result.library, result.fndesc.llvm_func_name
    kernel_fixup(library.get_function(entry), options["debug"])
    nvvm.set_launch_bounds(library.get_function(entry), options["launch_bounds"])
    if library.use_cooperative:
        raise TypeError(f"{function.__qualname__} syncs its grid, which patos cannot launch")
    return library, entry


def _linked(library: CUDACodeLibrary, options: Mapping) -> ObjectCode:
    """The cubin of `library` for the current GPU, linked with every unit it names."""
    major, minor = cuda.get_current_device().compute_capability
    target = (major, minor, "a" if (major, minor) >= (9, 0) else "")
    compiled = _nvvm_options(options) | {"arch": nvrtc.get_arch_option(*target)}
    linker = driver._Linker(
        max_registers=options.get("max_registers"), cc=target, lto=options["lto"]
    )
    if options["lto"]:
        linker.add_ltoir(_nvvm(library.llvm_strs, compiled | {"gen-lto": None}))
    else:
        for code in _separated(library.llvm_strs, compiled):
            linker.add_ptx(code.decode().strip("\x00").strip().encode())
    for path in (*options["link"], *_needed(library), *library._linking_files):
        linker.add_file_guess_ext(path)
    return linker.complete()


def _nvvm_options(options: Mapping) -> dict[str, int | bool | None]:
    """The NVVM options numba-cuda's `_Kernel` compiles at: fastmath, the level, debug info."""
    debug = {"g": None} if options["debug"] else {}
    return {"fastmath": options["fastmath"], "opt": 3 if options["opt"] else 0} | debug


def _separated(modules: Sequence[str], options: Mapping) -> list[bytes]:
    """The PTX of `modules`: one compile, or one per module where debug info is asked for."""
    if "g" in options:
        return [_nvvm([module], options) for module in modules]
    return [_nvvm(modules, options)]


def _nvvm(modules: Sequence[str], options: Mapping) -> bytes:
    """What `nvvm.compile_ir` returns for `modules`, without the verifier's pass."""
    options = dict(options)
    if options.pop("fastmath", False):
        options |= {"ftz": True, "fma": True, "prec_div": False, "prec_sqrt": False}
    unit = nvvm.CompilationUnit(options)
    for module in modules:
        unit.add_module(nvvm.llvm_replace(module).encode())
    unit.lazy_add_module(nvvm.LibDevice().get())
    return unit.compile()


def _needed(library: CUDACodeLibrary) -> tuple[str, ...]:
    """The numba-cuda helper library the kernel's IR calls into, when it does."""
    code = library.get_llvm_str()
    return (str(_RESHAPING),) if any(name in code for name in dispatcher.reshape_funcs) else ()
