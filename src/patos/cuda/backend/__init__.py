"""The one place patos.cuda uses numba-cuda's and llvmlite's internals: the typing and lowering of
records, lanes, named values, PTX and C++ intrinsics and dispatched names, the C integer rules, the
`device` and `kernel` compiles, and the marshalling of a launch's arguments into Numba's layouts.

The internals it reaches are numba-cuda 0.30's typing and lowering registries (`cudadecl`,
`cudaimpl`), `typing.templates`, `extending`, `cgutils`, `core.imputils`, `CUDADispatcher` (device
functions), `compiler.compile_cuda` and `kernel_fixup`, `cudadrv.nvvm`, `cudadrv.nvrtc` and the
driver's `_Linker` (kernels), `CUDACodeLibrary`'s `llvm_strs`, `linking_libraries`,
`referenced_objects` and `_linking_files`, the array model's field order, and llvmlite's IR
builder. A numba-cuda upgrade is checked against this package alone.
"""
