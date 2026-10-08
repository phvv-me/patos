"""Launch one Numba kernel with its arguments marshalled once.

numba-cuda hands every launch to cuda.core's `launch()`; what costs is the typing and
descriptor building above it, so the descriptors are kept beside the compiled kernel and
rebuilt only for an argument that changed.
"""

from collections.abc import Hashable, Sequence
from typing import NamedTuple

import cupy as cp
from cuda.core import LaunchConfig, Stream, launch
from cuda.core.utils import StridedMemoryView
from cupy.cuda import get_current_stream
from numba.cuda.np.numpy_support import map_layout

type Kind = tuple[type, str, int, str] | tuple[type, tuple[Kind, ...]] | type
type Identity = (
    tuple[int, tuple[int, ...], tuple[int, ...]] | tuple[type, Hashable] | tuple[Identity, ...]
)


class Grid(NamedTuple):
    """One launch's shape: how many blocks, and how many threads in each."""

    blocks: int
    threads: int


class Specialization:
    """One compiled kernel with the descriptors of the arguments it was last launched on."""

    def __init__(self, kernel, count: int) -> None:
        self.kernel = kernel
        # Numba has no public door to a kernel's cuda.core handle or its marshalling: numba-cuda
        # 0.30 keeps no public handle to a compiled kernel's cuda.core kernel (2026-09-10).
        self.core_kernel = kernel._codelibrary.get_cufunc().kernel
        self.prepared: list[tuple[Identity, list] | None] = [None] * count

    def arguments(self, stream, arguments: tuple) -> list:
        """Marshal the arguments, rebuilding only the descriptors that no longer match."""
        kernelargs: list = []
        for index, (kind, value) in enumerate(
            zip(self.kernel.argument_types, arguments, strict=True)
        ):
            identity = self._identity(value)
            slot = self.prepared[index]
            if slot is None or slot[0] != identity:
                slot = self.prepared[index] = (identity, self._marshalled(kind, value, stream))
            kernelargs.extend(slot[1])
        return kernelargs

    @staticmethod
    def _identity(value) -> Identity:
        """What decides whether a marshalled argument still describes `value`.

        A device array is its pointer, shape and strides, a scalar its own value, a tuple the
        identities of its elements.
        """
        if isinstance(value, Sequence):
            return tuple(Specialization._identity(item) for item in value)
        data = getattr(value, "data", None)
        if data is not None and hasattr(data, "ptr"):
            return (data.ptr, value.shape, value.strides)
        return (type(value), value)

    def _marshalled(self, kind, value, stream) -> list:
        """The kernel-argument descriptors Numba builds for one argument."""
        marshalled: list = []
        # numba-cuda 0.30 keeps no public argument marshalling entry (2026-09-10).
        self.kernel._prepare_args(kind, self._viewed(value, stream), stream, [], marshalled)
        return marshalled

    def _viewed(self, value, stream):
        """`value`, or a strided view of it when CuPy would export its negative strides wrongly."""
        # CuPy 14.2.0's DLPack export corrupts negative strides (upstream #10228).
        if not (
            type(value) is cp.ndarray
            and value.size
            and not self.kernel.extensions
            and any(stride < 0 for stride in value.strides)
        ):
            return value
        try:
            return StridedMemoryView.from_cuda_array_interface(
                value, stream_ptr=int(stream.handle) or 1
            )
        except BufferError:
            if value.__cuda_array_interface__["version"] >= 3:
                raise
            return value


class CachedCudaLauncher:
    """Launch one Numba kernel through cuda.core with cached argument descriptors.

    Numba compiles each signature; all launches marshal only arguments that changed.
    The current CuPy stream declares the producer, including on descriptor-cache hits.
    """

    def __init__(self, dispatcher, *, enabled: bool = True) -> None:
        self.dispatcher = dispatcher
        self.enabled = enabled
        self.specializations: dict[tuple[Kind, ...], Specialization] = {}
        self.configs: dict[Grid, LaunchConfig] = {}
        self.streams: dict[int, Stream] = {}

    @property
    def compiled(self) -> bool:
        """Whether this launcher has already specialized a call in this process."""
        return bool(self.specializations)

    @classmethod
    def each(cls, kernels, *, enabled: bool) -> tuple[CachedCudaLauncher, ...]:
        """One launcher per kernel, in the order given."""
        return tuple(cls(kernel, enabled=enabled) for kernel in kernels)

    def launch(self, grid: Grid, stream, *arguments) -> None:
        """Launch with the cached kernel and descriptors of this argument signature.

        stream: any stream speaking the CUDA stream protocol, CuPy's, numba's or cuda.core's.
        """
        consumer = self._core_stream(stream)
        if not self.enabled:
            self.dispatcher[grid.blocks, grid.threads, consumer](*arguments)
            return
        signature = tuple(self._kind(value) for value in arguments)
        specialization = self.specializations.get(signature)
        if specialization is None:
            specialized = self.dispatcher.specialize(*arguments)
            kernel = next(iter(specialized.overloads.values()))
            specialization = self.specializations[signature] = Specialization(
                kernel, len(arguments)
            )
        config = self.configs.get(grid)
        if config is None:
            config = LaunchConfig(grid=grid.blocks, block=grid.threads)
            self.configs[grid] = config
        producer = get_current_stream()
        if (int(consumer.handle) or 1) != (producer.ptr or 1):
            consumer.wait(producer)
        launch(
            consumer,
            config,
            specialization.core_kernel,
            *specialization.arguments(consumer, arguments),
        )

    @classmethod
    def _kind(cls, value) -> Kind:
        """What decides which compiled specialization an argument belongs to.

        Array layout selects different compiled indexing; scalars also keep their type.
        """
        if isinstance(value, tuple):
            return (type(value), tuple(cls._kind(item) for item in value))
        dtype = getattr(value, "dtype", None)
        if dtype is not None:
            return (type(value), dtype.str, value.ndim, map_layout(value))
        return type(value)

    def _core_stream(self, stream) -> Stream:
        """Borrow `stream` as a cuda.core stream, wrapped once per handle.

        This is what Numba's ordinary launch path does; the caller still owns the stream and
        marshalling keeps the wrapper.
        """
        if isinstance(stream, Stream):
            return stream
        handle = int(stream.__cuda_stream__()[1])
        core_stream = self.streams.get(handle)
        if core_stream is None:
            core_stream = self.streams[handle] = Stream.from_handle(handle)
        return core_stream


_launchers: dict[Hashable, CachedCudaLauncher] = {}


def launch_kernel(kernel, grid: Grid, stream, *arguments) -> None:
    """Launch `kernel` through the one cached launcher every call site shares for it.

    The launcher is keyed on the dispatcher object itself.
    """
    launcher = _launchers.get(kernel)
    if launcher is None:
        launcher = CachedCudaLauncher(kernel)
        _launchers[kernel] = launcher
    launcher.launch(grid, stream, *arguments)
