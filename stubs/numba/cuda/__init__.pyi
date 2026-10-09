"""numba-cuda 0.30's `numba.cuda` as a kernel reads it, for the type checkers alone.

numba declares its intrinsics as stub classes and compile-time `@intrinsic` objects, so a checker
reads `cuda.threadIdx.x` as a `property` and `cuda.syncthreads()` as missing its typing context.
Here each takes and returns what its compiled call does, in patos's numeric types: a scalar is an
`int` or a `float`, and an array is a `Numeric`, which device code spells `Vector` or `Matrix`, so
`shared.array(n, i32)` is a `Vector[i32]`. A dtype is the element's class (`i32`, `np.uint64`,
`type(value)`) or a numba type (`numba.float32`). The vector types (`float32x4`, ...) numba sets on
the module at import are left out. The host API is numba's own, re-exported as its modules type it.
"""

from collections.abc import Callable, Sequence
from typing import Final, Literal, Never, Protocol, TypedDict, Unpack, overload, type_check_only

import numpy as np
from numba.core.types import Type
from numba.core.typing import Signature

from patos.cuda.scalars import Numeric

from . import cg as cg
from .api import as_cuda_array as as_cuda_array
from .api import close as close
from .api import current_context as current_context
from .api import default_stream as default_stream
from .api import defer_cleanup as defer_cleanup
from .api import detect as detect
from .api import device_array as device_array
from .api import device_array_like as device_array_like
from .api import event as event
from .api import event_elapsed_time as event_elapsed_time
from .api import external_stream as external_stream
from .api import from_cuda_array_interface as from_cuda_array_interface
from .api import get_current_device as get_current_device
from .api import gpus as gpus
from .api import is_bfloat16_supported as is_bfloat16_supported
from .api import is_cuda_array as is_cuda_array
from .api import is_float16_supported as is_float16_supported
from .api import is_fp8_supported as is_fp8_supported
from .api import legacy_default_stream as legacy_default_stream
from .api import list_devices as list_devices
from .api import managed_array as managed_array
from .api import mapped as mapped
from .api import mapped_array as mapped_array
from .api import mapped_array_like as mapped_array_like
from .api import open_ipc_array as open_ipc_array
from .api import per_thread_default_stream as per_thread_default_stream
from .api import pinned as pinned
from .api import pinned_array as pinned_array
from .api import pinned_array_like as pinned_array_like
from .api import profile_start as profile_start
from .api import profile_stop as profile_stop
from .api import profiling as profiling
from .api import require_context as require_context
from .api import select_device as select_device
from .api import stream as stream
from .api import synchronize as synchronize
from .api import to_device as to_device
from .args import In as In
from .args import InOut as InOut  # codespell:ignore inout
from .args import Out as Out
from .compiler import compile as compile
from .compiler import compile_for_current_device as compile_for_current_device
from .compiler import compile_ptx as compile_ptx
from .compiler import compile_ptx_for_current_device as compile_ptx_for_current_device
from .cudadrv import nvvm as nvvm
from .cudadrv.driver import BaseCUDAMemoryManager as BaseCUDAMemoryManager
from .cudadrv.driver import GetIpcHandleMixin as GetIpcHandleMixin
from .cudadrv.driver import HostOnlyCUDAMemoryManager as HostOnlyCUDAMemoryManager
from .cudadrv.driver import IpcHandle as IpcHandle
from .cudadrv.driver import MappedMemory as MappedMemory
from .cudadrv.driver import MemoryInfo as MemoryInfo
from .cudadrv.driver import MemoryPointer as MemoryPointer
from .cudadrv.driver import PinnedMemory as PinnedMemory
from .cudadrv.driver import set_memory_manager as set_memory_manager
from .cudadrv.error import CudaSupportError as CudaSupportError
from .cudadrv.runtime import runtime as runtime
from .decorators import declare_device as declare_device
from .dispatcher import CUDADispatcher
from .errors import KernelRuntimeError as KernelRuntimeError
from .kernels.reduction import Reduce as Reduce

type _Index = int | tuple[int, ...]
type _Signature = Signature | tuple[Type | Signature, ...] | str

implementation: Final[str]
reduce = Reduce

def is_available() -> bool: ...
def is_supported_version() -> bool: ...
def cuda_error() -> str | None: ...

@type_check_only
class _Dim3:
    """The index or extent of a thread or block on each axis."""

    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def z(self) -> int: ...

threadIdx: Final[_Dim3]
blockIdx: Final[_Dim3]
blockDim: Final[_Dim3]
gridDim: Final[_Dim3]
laneid: Final[int]
warpsize: Final[int]

@overload
def grid(ndim: Literal[1]) -> int: ...
@overload
def grid(ndim: Literal[2]) -> tuple[int, int]: ...
@overload
def grid(ndim: Literal[3]) -> tuple[int, int, int]: ...
@overload
def gridsize(ndim: Literal[1]) -> int: ...
@overload
def gridsize(ndim: Literal[2]) -> tuple[int, int]: ...
@overload
def gridsize(ndim: Literal[3]) -> tuple[int, int, int]: ...

# What an array holds: the numbers of Python and of numpy.
type _Element = int | float | complex | np.number | np.bool_

@type_check_only
class _NumbaType[T](Protocol):
    """A numba type (`numba.float32`), which casts a value to the element `T`."""

    def cast_python_value(self, value: Never, /) -> T: ...

@type_check_only
class _MemorySpace:
    """`shared` or `local`: an array of a constant shape, one per block or one per thread.

    The element is what calling the dtype makes (`i32(0)` an `int`, `np.uint64(0)` a `np.uint64`)
    and not a `type[T]` parameter. pyrefly reads `type(value)` of a type parameter as the bare
    `type`, which `type[T]` solves from the array's first use: `Vector[number]` there makes `T` a
    `float`. Called, the bare `type` answers `Any`, so pyrefly leaves that element untyped and ty
    reads it as `T`. A numba type is called to make a signature, so it is read apart, by its cast.
    """

    @overload
    def array[T: _Element](
        self, shape: int | tuple[int], dtype: _NumbaType[T], alignment: int | None = None
    ) -> Numeric[T, int]: ...
    @overload
    def array[T: _Element](
        self, shape: int | tuple[int], dtype: Callable[..., T], alignment: int | None = None
    ) -> Numeric[T, int]: ...
    @overload
    def array[T: _Element](
        self, shape: tuple[int, int], dtype: _NumbaType[T], alignment: int | None = None
    ) -> Numeric[T, int, int]: ...
    @overload
    def array[T: _Element](
        self, shape: tuple[int, int], dtype: Callable[..., T], alignment: int | None = None
    ) -> Numeric[T, int, int]: ...
    @overload
    def array[T: _Element](
        self, shape: tuple[int, int, int], dtype: _NumbaType[T], alignment: int | None = None
    ) -> Numeric[T, int, int, int]: ...
    @overload
    def array[T: _Element](
        self, shape: tuple[int, int, int], dtype: Callable[..., T], alignment: int | None = None
    ) -> Numeric[T, int, int, int]: ...
    @overload
    def array[T: _Element](
        self, shape: tuple[int, ...], dtype: _NumbaType[T], alignment: int | None = None
    ) -> Numeric[T, *tuple[int, ...]]: ...
    @overload
    def array[T: _Element](
        self, shape: tuple[int, ...], dtype: Callable[..., T], alignment: int | None = None
    ) -> Numeric[T, *tuple[int, ...]]: ...

@type_check_only
class _ConstantSpace:
    def array_like[A](self, ndarray: A) -> A: ...

shared: Final[_MemorySpace]
local: Final[_MemorySpace]
const: Final[_ConstantSpace]

@type_check_only
class _Arithmetic:
    def add[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...
    def dec[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...
    def inc[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...
    def sub[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...

@type_check_only
class _Extremes:
    def max[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...
    def min[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...
    def nanmax[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...
    def nanmin[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...

@type_check_only
class _Bitwise:
    def and_[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...
    def or_[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...
    def xor[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...

@type_check_only
class _Exchange:
    def cas[T, *S](self, ary: Numeric[T, *S], idx: _Index, old: T, val: T) -> T: ...
    def compare_and_swap[T](self, ary: Numeric[T, int], old: T, val: T) -> T: ...
    def exch[T, *S](self, ary: Numeric[T, *S], idx: _Index, val: T) -> T: ...

@type_check_only
class _Atomics(_Arithmetic, _Extremes, _Bitwise, _Exchange):
    """Each updates `ary[idx]` in one indivisible step and returns the element it replaced."""

atomic: Final[_Atomics]

def syncthreads() -> None: ...
def syncthreads_count(predicate: int) -> int: ...
def syncthreads_and(predicate: int) -> int: ...
def syncthreads_or(predicate: int) -> int: ...
def syncwarp(mask: int = 0xFFFFFFFF) -> None: ...
def threadfence() -> None: ...
def threadfence_block() -> None: ...
def threadfence_system() -> None: ...
def nanosleep(ns: int) -> None: ...

# A shuffle returns the `value` another lane of `mask` holds, of `value`'s own type.
def shfl_sync[T: (int, float)](mask: int, value: T, src_lane: int) -> T: ...
def shfl_up_sync[T: (int, float)](mask: int, value: T, delta: int) -> T: ...
def shfl_down_sync[T: (int, float)](mask: int, value: T, delta: int) -> T: ...
def shfl_xor_sync[T: (int, float)](mask: int, value: T, lane_mask: int) -> T: ...
def all_sync(mask: int, predicate: int) -> bool: ...
def any_sync(mask: int, predicate: int) -> bool: ...
def eq_sync(mask: int, predicate: int) -> bool: ...
def ballot_sync(mask: int, predicate: int) -> int: ...
def match_any_sync(mask: int, value: float) -> int: ...
def match_all_sync(mask: int, value: float) -> tuple[int, bool]: ...
def activemask() -> int: ...
def lanemask_lt() -> int: ...
def popc(x: int) -> int: ...
def brev(x: int) -> int: ...
def clz(x: int) -> int: ...
def ffs(x: int) -> int: ...
def selp[T: (int, float)](test: int, a: T, b: T) -> T: ...
def fma(a: float, b: float, c: float) -> float: ...
def cbrt(a: float) -> float: ...

# Loads and stores of `array[i]` through the cache operator each names.
def ldca[T, *S](array: Numeric[T, *S], i: _Index) -> T: ...
def ldcg[T, *S](array: Numeric[T, *S], i: _Index) -> T: ...
def ldcs[T, *S](array: Numeric[T, *S], i: _Index) -> T: ...
def ldlu[T, *S](array: Numeric[T, *S], i: _Index) -> T: ...
def ldcv[T, *S](array: Numeric[T, *S], i: _Index) -> T: ...
def stcg[T, *S](array: Numeric[T, *S], i: _Index, value: T) -> None: ...
def stcs[T, *S](array: Numeric[T, *S], i: _Index, value: T) -> None: ...
def stwb[T, *S](array: Numeric[T, *S], i: _Index, value: T) -> None: ...
def stwt[T, *S](array: Numeric[T, *S], i: _Index, value: T) -> None: ...

@type_check_only
class _Jitted[**P, R](CUDADispatcher):
    """What `jit` returns: device code calls it as the function it compiled."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

@type_check_only
class _Jit(Protocol):
    def __call__[**P, R](self, function: Callable[P, R], /) -> _Jitted[P, R]: ...

@type_check_only
class _JitOptions(TypedDict, total=False):
    device: bool
    inline: Literal["never", "always"] | bool
    forceinline: bool
    link: Sequence[str]
    debug: bool | None
    opt: bool | None
    lineinfo: bool
    cache: bool
    launch_bounds: int | tuple[int, ...] | None
    lto: bool | None
    shared_memory_carveout: Literal["MaxL1", "MaxShared", "default"] | int | None
    fastmath: bool
    max_registers: int

@overload
def jit[**P, R](func_or_sig: Callable[P, R], **options: Unpack[_JitOptions]) -> _Jitted[P, R]: ...
@overload
def jit[S: _Signature](
    func_or_sig: S | list[S] | None = None, **options: Unpack[_JitOptions]
) -> _Jit: ...
