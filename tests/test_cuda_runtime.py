"""The CUDA host runtime, tested without a GPU.

`Workspace`, `pinned` and the stream cache run on numpy and plain fakes. The launcher and the
`cuda.compute` wrappers import against stand-ins for CuPy, cuda.core and cuda.compute, which a
CPU-only machine lacks, so what is tested is their own caching and ordering. Only the last test
needs a real CUDA stack and skips without one.
"""

import importlib
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import combinations
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

_DESCRIPTOR = "descriptor"
dtypes = st.sampled_from([np.uint8, np.int32, np.float64])
takes = st.lists(
    st.tuples(st.sampled_from(["scan", "keys", "out"]), st.integers(0, 40), dtypes),
    min_size=1,
    max_size=30,
)


class FakeStream:
    """A cupy, cuda.core or numba stream reduced to a handle and a record of its use."""

    def __init__(self, handle: int = 0) -> None:
        self.handle = self.ptr = handle
        self.waited: list[FakeStream] = []
        self.synchronized = 0

    @classmethod
    def from_handle(cls, handle: int) -> FakeStream:
        return cls(handle)

    def synchronize(self) -> None:
        self.synchronized += 1

    def wait(self, other: FakeStream) -> None:
        self.waited.append(other)


class FakeDeviceArray:
    """What the launcher reads of a `cupy.ndarray`: a pointer, a shape and strides, a layout."""

    def __init__(
        self,
        pointer: int,
        shape: Sequence[int] = (4,),
        *,
        strides: Sequence[int] = (4,),
        version=3,
    ) -> None:
        self.data = SimpleNamespace(ptr=pointer)
        self.shape = shape
        self.strides = strides
        self.size = int(np.prod(shape))
        self.ndim = len(shape)
        self.dtype = np.dtype(np.int32)
        self.flags = SimpleNamespace(c_contiguous=True, f_contiguous=True)
        self.__cuda_array_interface__ = {"version": version}


class FakeKernel:
    """A compiled Numba kernel: argument types, a cuda.core handle and a marshalling entry."""

    def __init__(self, *argument_types: str, extensions: Sequence[str] = ()) -> None:
        self.argument_types = argument_types
        self.extensions = extensions
        self.core = SimpleNamespace()
        self.prepared: list[tuple[str, object]] = []
        self._codelibrary = SimpleNamespace(get_cufunc=lambda: SimpleNamespace(kernel=self.core))

    def _prepare_args(self, kind: str, value, stream, _retained, marshalled: list) -> None:
        self.prepared.append((kind, value))
        marshalled.append((_DESCRIPTOR, kind, value))


class FakeDispatcher:
    """A Numba dispatcher that specializes to one kernel and records direct launches."""

    def __init__(self, kernel: FakeKernel) -> None:
        self.kernel = kernel
        self.specialized: list[Sequence] = []
        self.direct: list[Sequence] = []

    def __getitem__(self, configuration: Sequence):
        return lambda *arguments: self.direct.append((configuration, arguments))

    def specialize(self, *arguments) -> SimpleNamespace:
        self.specialized.append(arguments)
        return SimpleNamespace(overloads={"signature": self.kernel})


class FakeAlgorithm:
    """A built `cuda.compute` algorithm: a temporary-storage query, then a run."""

    def __init__(self, factory: str, built_with: Mapping, state: SimpleNamespace) -> None:
        self.factory = factory
        self.built_with = built_with
        self.state = state
        self.calls: list[Mapping] = []

    def __call__(self, **call) -> int | None:
        self.calls.append(call)
        return self.state.needed if call["temp_storage"] is None else None


def fake_libraries(state: SimpleNamespace) -> dict[str, ModuleType]:
    """CuPy, cuda.core, cuda.compute and Numba's layout map, as `patos.cuda` imports them."""

    def module(name: str, **attributes) -> ModuleType:
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        return stub

    def strided_view(value: FakeDeviceArray, stream_ptr: int):
        if state.export_error:
            raise state.export_error
        return ("view", value, stream_ptr)

    def factory(name: str):
        def make(**built_with) -> FakeAlgorithm:
            algorithm = FakeAlgorithm(name, built_with, state)
            state.builds.append(algorithm)
            return algorithm

        return make

    def map_layout(array) -> str:
        return "C" if array.flags.c_contiguous else "F" if array.flags.f_contiguous else "A"

    core = module(
        "cuda.core",
        LaunchConfig=lambda grid, block: SimpleNamespace(grid=grid, block=block),
        Stream=FakeStream,
        launch=lambda *arguments: state.launches.append(arguments),
    )
    compute = module(
        "cuda.compute",
        make_exclusive_scan=factory("make_exclusive_scan"),
    )
    return {
        "cupy": module("cupy", ndarray=FakeDeviceArray),
        "cupy.cuda": module("cupy.cuda", get_current_stream=lambda: state.current),
        "cuda": module("cuda", core=core, compute=compute),
        "cuda.core": core,
        "cuda.core.utils": module(
            "cuda.core.utils",
            StridedMemoryView=SimpleNamespace(from_cuda_array_interface=strided_view),
        ),
        "cuda.compute": compute,
        "cuda.compute.iterators": module("cuda.compute.iterators", IteratorBase=type),
        "cuda.compute.typing": module("cuda.compute.typing", DeviceArrayLike=type),
        "numba.cuda.np.numpy_support": module(
            "numba.cuda.np.numpy_support", map_layout=map_layout
        ),
    }


@contextmanager
def fake_runtime() -> Iterator[SimpleNamespace]:
    """The `patos.cuda.runtime` modules freshly imported against `fake_libraries`.

    Yields the shared fake state, with the modules as `launch`, `memory` and `streams`. Every
    `patos.cuda` module imported meanwhile is dropped on exit, so no later test meets a module
    bound to a stand-in.
    """
    state = SimpleNamespace(
        current=FakeStream(1), launches=[], builds=[], export_error=None, needed=0
    )
    package = importlib.import_module("patos.cuda")
    with pytest.MonkeyPatch.context() as patch:
        for name, stub in fake_libraries(state).items():
            patch.setitem(sys.modules, name, stub)
        for name in [name for name in sys.modules if name.startswith("patos.cuda.")]:
            patch.delitem(sys.modules, name)
            patch.delattr(package, name.split(".")[2], raising=False)
        try:
            state.launch = importlib.import_module("patos.cuda.runtime.launch")
            state.memory = importlib.import_module("patos.cuda.runtime.memory")
            state.streams = importlib.import_module("patos.cuda.runtime.streams")
            yield state
        finally:
            for name in [name for name in sys.modules if name.startswith("patos.cuda.")]:
                del sys.modules[name]
                package.__dict__.pop(name.split(".")[2], None)


@pytest.fixture
def runtime() -> Iterator[SimpleNamespace]:
    """The fake runtime of `fake_runtime`, for tests that need no Hypothesis."""
    with fake_runtime() as state:
        yield state


@pytest.fixture(scope="module")
def memory() -> ModuleType:
    """`patos.cuda.runtime.memory`, imported under the fakes like the launcher beside it."""
    with fake_runtime() as state:
        return state.memory


@pytest.fixture(scope="module")
def streams() -> ModuleType:
    """`patos.cuda.runtime.streams`, imported under the fakes like the launcher beside it."""
    with fake_runtime() as state:
        return state.streams


@given(wanted=takes)
def test_workspace_serves_exact_views_over_buffers_that_grow_and_never_alias(
    memory: ModuleType, wanted
) -> None:
    """A role is reallocated only to grow or to change dtype, and no two roles share storage."""
    workspace = memory.Workspace(np)

    for role, size, dtype in wanted:
        before = workspace.buffers.get(role)
        view = workspace.take(role, size, dtype)
        held = workspace.buffers[role]

        reused = before is not None and before.dtype == dtype and before.shape[0] >= size
        assert view.shape == (size,)
        assert view.dtype == dtype
        assert size == 0 or np.shares_memory(view, held)
        assert (held is before) == reused
        assert reused or held.shape[0] == max(size, 1)

    assert list(workspace) == list(dict.fromkeys(role for role, _, _ in wanted))
    assert not any(
        np.shares_memory(first, second)
        for first, second in combinations(workspace.buffers.values(), 2)
    )


@given(size=st.integers(0, 40), dtype=dtypes, junk=st.integers(1, 100))
def test_zeros_clears_what_an_earlier_take_left(
    memory: ModuleType, size: int, dtype, junk: int
) -> None:
    """The scratch is reused as is, so only `zeros` promises its contents."""
    workspace = memory.Workspace(np)
    workspace.take("buffer", size, dtype).fill(junk)

    assert not np.asarray(workspace.zeros("buffer", size, dtype)).any()
    assert not np.asarray(workspace.buffers["buffer"][:size]).any()


@given(sizes=st.tuples(st.integers(1, 20), st.integers(21, 40)))
def test_retain_replaced_keeps_replaced_buffers_until_the_outermost_scope_closes(
    memory: ModuleType, sizes
) -> None:
    """Growing or retyping a role inside a scope parks its old buffer until the last exit."""
    workspace = memory.Workspace(np)
    workspace.take("buffer", sizes[0], np.int32)
    original = workspace.buffers["buffer"]

    with workspace.retain_replaced():
        with workspace.retain_replaced():
            workspace.take("buffer", sizes[1], np.int32)
            grown = workspace.buffers["buffer"]
            assert [id(buffer) for buffer in workspace.retained] == [id(original)]
            workspace.take("buffer", sizes[1], np.float64)
        assert [id(buffer) for buffer in workspace.retained] == [id(original), id(grown)]

    workspace.take("buffer", sizes[1] + 1, np.uint8)
    assert (workspace.retained, workspace.retention_depth) == ([], 0)


def test_settle_uploads_synchronizes_the_current_stream_and_skips_modules_without_one(
    streams,
) -> None:
    """An array module with no CUDA streams, such as numpy, has nothing to settle."""
    stream = FakeStream()
    arrays = SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda: stream))

    streams.settle_uploads(arrays)
    streams.settle_uploads(np)

    assert stream.synchronized == 1


def test_kind_separates_exactly_what_numba_compiles_differently() -> None:
    """Dtype, rank and layout pick a specialization; values do not, and tuples nest."""
    specs = st.tuples(
        st.sampled_from([np.int32, np.float32, np.uint8]), st.integers(1, 2), st.sampled_from("CF")
    )

    def zeros_for(spec: Sequence) -> np.ndarray:
        """A zero array with shape `[3, 4][:rank]`, of the spec's dtype and memory order."""
        dtype, rank, order = spec
        return np.zeros((3, 4)[:rank], dtype=dtype, order=order)

    def signature(spec: Sequence) -> Sequence:
        dtype, rank, order = spec
        return (np.dtype(dtype).str, rank, order if rank == 2 else "C")

    with fake_runtime() as state:
        kind = state.launch.CachedCudaLauncher._kind

        @given(left=specs, right=specs)
        def check(left: Sequence, right: Sequence) -> None:
            same_kind = kind(zeros_for(left)) == kind(zeros_for(right))
            assert same_kind == (signature(left) == signature(right))
            assert kind((zeros_for(left), 3)) == (tuple, (kind(zeros_for(left)), int))

        check()
        assert kind(3) is int
        assert kind(2.5) is float


def test_identity_is_the_pointer_shape_and_strides_of_arrays_and_the_value_of_scalars() -> None:
    """A marshalled descriptor stays valid exactly while its argument's identity does."""
    with fake_runtime() as state:
        identity = state.launch.Specialization._identity

        @given(
            pointer=st.integers(0, 2**40),
            shape=st.tuples(st.integers(1, 8), st.integers(1, 8)),
            scalar=st.integers(),
        )
        def check(pointer: int, shape: Sequence[int], scalar: int) -> None:
            strides = (shape[1] * 4, 4)
            array = FakeDeviceArray(pointer, shape, strides=strides)
            assert identity(array) == (pointer, shape, strides)
            assert identity(FakeDeviceArray(pointer + 1, shape, strides=strides)) != identity(
                array
            )
            assert identity(scalar) == (int, scalar)
            assert identity(float(scalar)) != identity(scalar)
            assert identity((array, scalar)) == (identity(array), identity(scalar))

        check()


def test_arguments_marshal_only_what_changed(runtime: SimpleNamespace) -> None:
    """A repeat reuses every descriptor; a moved array or a new scalar rebuilds only its own."""
    kernel = FakeKernel("array", "scalar")
    specialization = runtime.launch.Specialization(kernel, 2)
    stream = FakeStream(5)
    array = FakeDeviceArray(100)

    first = specialization.arguments(stream, (array, 3))
    assert specialization.core_kernel is kernel.core
    assert first == [(_DESCRIPTOR, "array", array), (_DESCRIPTOR, "scalar", 3)]
    assert specialization.arguments(stream, (array, 3)) == first
    assert len(kernel.prepared) == 2

    specialization.arguments(stream, (array, 4))
    assert kernel.prepared[2:] == [("scalar", 4)]
    specialization.arguments(stream, (FakeDeviceArray(200), 4))
    assert [kind for kind, _ in kernel.prepared[3:]] == ["array"]


@pytest.mark.parametrize(
    ("array", "extensions", "export_error", "route"),
    [
        pytest.param(FakeDeviceArray(1, strides=(-4,)), (), None, "view", id="negative-viewed"),
        pytest.param(FakeDeviceArray(1, strides=(4,)), (), None, "raw", id="positive-passed"),
        pytest.param(FakeDeviceArray(1, (0,), strides=(-4,)), (), None, "raw", id="empty-passed"),
        pytest.param(FakeDeviceArray(1, strides=(-4,)), ("ext",), None, "raw", id="extensions"),
        pytest.param(
            FakeDeviceArray(1, strides=(-4,), version=2),
            (),
            BufferError,
            "raw",
            id="old-falls-back",
        ),
        pytest.param(
            FakeDeviceArray(1, strides=(-4,)), (), BufferError, "raises", id="new-reraises"
        ),
    ],
)
def test_negative_strides_are_viewed_around_cupy_dlpack_export(
    runtime: SimpleNamespace,
    array: FakeDeviceArray,
    extensions: Sequence[str],
    export_error: type[BufferError] | None,
    route: str,
) -> None:
    """CuPy 14.2.0 corrupts negative strides through DLPack, so those go in as strided views."""
    runtime.export_error = export_error
    kernel = FakeKernel("array", extensions=extensions)
    specialization = runtime.launch.Specialization(kernel, 1)

    if route == "raises":
        with pytest.raises(BufferError):
            specialization.arguments(FakeStream(5), (array,))
        return
    specialization.arguments(FakeStream(5), (array,))

    assert kernel.prepared == [("array", ("view", array, 5) if route == "view" else array)]


def test_launcher_specializes_once_per_signature_and_shares_configs_and_streams(
    runtime: SimpleNamespace,
) -> None:
    """Typing, the launch config and the stream wrapper are built on first use, then shared."""
    kernel = FakeKernel("array", "scalar")
    dispatcher = FakeDispatcher(kernel)
    launcher = runtime.launch.CachedCudaLauncher(dispatcher)
    grid = runtime.launch.Grid(4, 128)
    stream = SimpleNamespace(__cuda_stream__=lambda: (0, 9))
    array = FakeDeviceArray(100)
    runtime.current = FakeStream(9)

    assert not launcher.compiled
    launcher.launch(grid, stream, array, 3)
    launcher.launch(runtime.launch.Grid(4, 128), stream, array, 7)

    consumer, config, core_kernel, *arguments = runtime.launches[0]
    assert launcher.compiled
    assert len(dispatcher.specialized) == 1
    assert (config.grid, config.block, core_kernel) == (4, 128, kernel.core)
    assert (launcher.configs, launcher.streams) == ({grid: config}, {9: consumer})
    assert arguments == [(_DESCRIPTOR, "array", array), (_DESCRIPTOR, "scalar", 3)]
    assert [launch[:3] for launch in runtime.launches] == [(consumer, config, core_kernel)] * 2
    assert consumer.waited == []


def test_launcher_waits_on_the_producer_and_specializes_again_for_a_new_signature(
    runtime: SimpleNamespace,
) -> None:
    """The current CuPy stream is the producer, and a core stream passed in is used as is."""
    dispatcher = FakeDispatcher(FakeKernel("array", "scalar"))
    launcher = runtime.launch.CachedCudaLauncher(dispatcher)
    grid = runtime.launch.Grid(4, 128)
    stream = SimpleNamespace(__cuda_stream__=lambda: (0, 9))
    array = FakeDeviceArray(100)
    runtime.current = FakeStream(11)

    launcher.launch(grid, stream, array, 3)
    consumer = runtime.launches[0][0]
    assert consumer.waited == [runtime.current]

    launcher.launch(grid, stream, array, 2.5)
    assert len(dispatcher.specialized) == 2

    own = FakeStream(13)
    launcher.launch(grid, own, array, 3)
    assert (runtime.launches[-1][0], launcher.streams) == (own, {9: consumer})


def test_launcher_runs_the_dispatcher_directly_when_disabled_and_launch_kernel_shares_one(
    runtime: SimpleNamespace,
) -> None:
    """`enabled=False` bypasses the cache; `launch_kernel` keeps one launcher per dispatcher."""
    kernels = [FakeDispatcher(FakeKernel("scalar")) for _ in range(2)]
    grid = runtime.launch.Grid(2, 64)
    stream = SimpleNamespace(__cuda_stream__=lambda: (0, 9))

    plain = runtime.launch.CachedCudaLauncher.each(kernels, enabled=False)
    plain[1].launch(grid, stream, 5)
    assert [launcher.dispatcher for launcher in plain] == kernels
    assert kernels[1].direct == [((2, 64, plain[1].streams[9]), (5,))]
    assert (runtime.launches, kernels[1].specialized) == ([], [])

    runtime.launch.launch_kernel(kernels[0], grid, stream, 5)
    runtime.launch.launch_kernel(kernels[0], grid, stream, 6)
    assert list(runtime.launch._launchers) == [kernels[0]]
    assert (len(kernels[0].specialized), len(runtime.launches)) == (1, 2)
