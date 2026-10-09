"""The graph a capture is building, and the branches a pass adds to it.

`cudaGraphSetConditional` exists only on the device, so a kernel hands each conditional node its
value from a device flag.
"""

from collections.abc import Callable
from contextlib import ExitStack
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Self

from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

if TYPE_CHECKING:
    from cuda.core.graph import GraphBuilder

_SETTER = r"""
extern "C" __device__ __cudart_builtin__ void CUDARTAPI cudaGraphSetConditional(
    cudaGraphConditionalHandle handle, unsigned int value);

extern "C" __global__ void set_from_flag(cudaGraphConditionalHandle handle, const int *flag) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        cudaGraphSetConditional(handle, flag[0] != 0);
    }
}
"""

type Work = Callable[[], None]

_current: ContextVar[Recording | None] = ContextVar("recording", default=None)


class Recording:
    """The graph a capture is building, to which a pass adds branches.

    Work launched while a branch builds lands in that branch, so `when` takes the branch's work
    as functions run with the branch's stream current.
    """

    def __init__(self, arrays, builder: GraphBuilder) -> None:
        """Start with the capture's own builder as the one work lands in.

        arrays: the array module owning device memory, normally `cupy`.
        """
        self.arrays = arrays
        self.builders = [builder]
        self.setter = None
        self.entered: list[Token[Recording | None]] = []

    def __enter__(self) -> Self:
        """Make this the recording `branch` adds to."""
        self.entered.append(_current.set(self))
        return self

    def __exit__(self, *_exception) -> None:
        _current.reset(self.entered.pop())

    def when(self, flag, then: Work, otherwise: Work | None = None) -> None:
        """Run `then` where the device value `flag[0]` is nonzero, else `otherwise`.

        The device chooses when the recording replays.
        """
        builder = self.builders[-1]
        condition = builder.create_condition(default_value=0)
        launch(
            builder, LaunchConfig(grid=1, block=1), self._setter(), condition, int(flag.data.ptr)
        )
        if otherwise is None:
            self._build(builder.if_then(condition), then)
            return
        taken, left = builder.if_else(condition)
        self._build(taken, then)
        self._build(left, otherwise)

    def _build(self, branch: GraphBuilder, work: Work) -> None:
        """Record `work` into `branch`, on the stream the branch captures."""
        with ExitStack() as scope:
            branch.begin_building()
            scope.callback(branch.end_building)
            self.builders.append(branch)
            scope.callback(self.builders.pop)
            scope.enter_context(self.arrays.cuda.Stream.from_external(branch.stream))
            work()

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


def branch(flag, then: Work, otherwise: Work | None = None) -> None:
    """Run `then` where the device value `flag[0]` is nonzero, else `otherwise`.

    A pass run as usual reads the value and branches on the host. A pass being recorded cannot
    read it, so both functions are recorded under a conditional node that the device decides,
    and each must leave the same arrays written for what follows, as a pass that ran either would.
    """
    recording = _current.get()
    if recording is not None:
        recording.when(flag, then, otherwise)
    elif int(flag[0]):
        then()
    elif otherwise is not None:
        otherwise()
