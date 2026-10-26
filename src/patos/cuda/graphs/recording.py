"""The graph a capture is building, and the branches and signals a pass adds to it.

`cudaGraphSetConditional` exists only on the device, so a kernel hands each conditional node its
value from a device flag.
"""

from collections.abc import Callable
from contextlib import ExitStack
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Self

import numpy as np
from cuda.bindings import runtime as cudart
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

from ..runtime.streams import checked, handle

if TYPE_CHECKING:
    from cuda.core.graph import GraphBuilder

    from ..runtime.memory import Interfaced
    from ..runtime.streams import Event, StreamLike

_SETTER = r"""
extern "C" __device__ __cudart_builtin__ void CUDARTAPI cudaGraphSetConditional(
    cudaGraphConditionalHandle handle, unsigned int value);

extern "C" __global__ void set_from_flag(cudaGraphConditionalHandle handle, const int *flag) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        cudaGraphSetConditional(handle, flag[0] != 0);
    }
}
"""

# Work a branch runs, given the stream to queue it on.
type Work = Callable[[StreamLike], None]

_current: ContextVar[Recording | None] = ContextVar("recording", default=None)


class Recording:
    """The graph a capture is building, to which a pass adds branches.

    Work queued while a branch builds lands in that branch, so `when` runs the branch's work
    given the branch's stream, which is also the array module's current stream meanwhile.
    """

    def __enter__(self) -> Self:
        """Make this the recording `branch` adds to."""
        self.entered.append(_current.set(self))
        return self

    def __exit__(self, *_exception) -> None:
        _current.reset(self.entered.pop())

    def __init__(self, arrays, builder: GraphBuilder) -> None:
        """Start with the capture's own builder as the one work lands in.

        arrays: the array module owning device memory, normally `cupy`.
        """
        self.arrays = arrays
        self.builders = [builder]
        self.setter = None
        self.entered: list[Token[Recording | None]] = []

    def _build(self, branch: GraphBuilder, work: Work) -> None:
        """Record `work` into `branch`, on the stream the branch captures."""
        with ExitStack() as scope:
            branch.begin_building()
            scope.callback(branch.end_building)
            self.builders.append(branch)
            scope.callback(self.builders.pop)
            work(scope.enter_context(self.arrays.cuda.Stream.from_external(branch.stream)))

    def _setter(self):
        """The kernel that sets a condition from a device flag, compiled for the device once."""
        if self.setter is None:
            parts = Device().compute_capability
            options = ProgramOptions(
                std="c++17", arch="sm_" + "".join(str(part) for part in parts)
            )
            module = Program(_SETTER, code_type="c++", options=options).compile(
                "cubin", name_expressions=("set_from_flag",)
            )
            self.setter = module.get_kernel("set_from_flag")
        return self.setter

    def when(self, flag: Interfaced, then: Work, otherwise: Work | None = None) -> None:
        """Run `then` where the device value `flag[0]` is nonzero, else `otherwise`.

        The device chooses when the recording replays.
        """
        builder = self.builders[-1]
        condition = builder.create_condition(default_value=0)
        address = flag.__cuda_array_interface__["data"][0]
        launch(builder, LaunchConfig(grid=1, block=1), self._setter(), condition, address)
        if otherwise is None:
            self._build(builder.if_then(condition), then)
            return
        taken, left = builder.if_else(condition)
        self._build(taken, then)
        self._build(left, otherwise)


def branch(
    flag: Interfaced, then: Work, otherwise: Work | None = None, *, stream: StreamLike
) -> None:
    """Run `then` where the device value `flag[0]` is nonzero, else `otherwise`, on `stream`.

    A pass run as usual reads the value once the work `stream` holds is done, and branches on the
    host. A pass being recorded cannot read it, so both functions are recorded under a conditional
    node that the device decides, each given its branch's stream, and each must leave the same
    arrays written for what follows, as a pass that ran either would.
    """
    recording = _current.get()
    if recording is not None:
        recording.when(flag, then, otherwise)
    elif _is_raised(flag, stream):
        then(stream)
    elif otherwise is not None:
        otherwise(stream)


def signal(event: Event, *, stream: StreamLike) -> None:
    """Record `event` on `stream`; in a recording, every replay signals it at this point.

    An event recorded into a graph the ordinary way cannot be waited for by the host after a
    replay. This one is an external event node, which can be, while the nodes after it still run.
    Outside a recording it is the plain record.
    """
    number = handle(stream)
    (status,) = checked(cudart.cudaStreamIsCapturing(number))
    active = status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    flags = cudart.cudaEventRecordExternal if active else cudart.cudaEventRecordDefault
    checked(cudart.cudaEventRecordWithFlags(event.ptr, number, flags))


def _is_raised(flag: Interfaced, stream: StreamLike) -> bool:
    """Whether `flag[0]` is nonzero once the work queued on `stream` is done."""
    interface = flag.__cuda_array_interface__
    held = np.empty(1, np.dtype(interface["typestr"]))
    source, number = interface["data"][0], handle(stream)
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    checked(cudart.cudaMemcpyAsync(held.ctypes.data, source, held.nbytes, kind, number))
    checked(cudart.cudaStreamSynchronize(number))
    return bool(held[0])
