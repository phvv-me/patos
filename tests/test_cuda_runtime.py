from contextlib import ExitStack
from itertools import combinations

import cupy as cp
import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from patos.cuda.runtime import Workspace, settle_uploads

roles = st.sampled_from(["scan", "keys", "out"])
dtypes = st.sampled_from([np.uint8, np.int32, np.float64])
on_gpu = pytest.mark.skipif(not cp.cuda.is_available(), reason="streams need a GPU")


class WorkspaceMachine(RuleBasedStateMachine):
    """Takes, zeroed takes and nested retention scopes in any order.

    A take is an exact view of its role's buffer, replaced only to grow or to change dtype; no two
    roles share storage; a zeroed take clears what an earlier one left; and every buffer replaced
    inside a scope stays alive until the outermost scope closes.
    """

    @invariant()
    def keeps_each_role_apart(self) -> None:
        buffers = self.workspace.buffers.values()
        assert list(self.workspace) == list(self.roles)
        assert not any(
            np.shares_memory(first, second) for first, second in combinations(buffers, 2)
        )

    @invariant()
    def retains_what_a_scope_replaced(self) -> None:
        assert [id(buffer) for buffer in self.workspace.retained] == list(map(id, self.replaced))
        assert self.workspace.retention_depth == len(self.scopes)

    @precondition(lambda machine: machine.scopes)
    @rule()
    def leave(self) -> None:
        self.scopes.pop().close()
        if not self.scopes:
            self.replaced.clear()

    @rule()
    def enter(self) -> None:
        scope = ExitStack()
        scope.enter_context(self.workspace.retain_replaced())
        self.scopes.append(scope)

    @rule(role=roles, size=st.integers(0, 40), dtype=dtypes, zeroed=st.booleans())
    def take(self, role: str, size: int, dtype: type[np.generic], *, zeroed: bool) -> None:
        before = self.workspace.buffers.get(role)
        generation = self.workspace.generation
        if zeroed:
            view = self.workspace.zeros(role, size, dtype, stream=cp.cuda.Stream.null)
        else:
            view = self.workspace.take(role, size, dtype)
        held = self.workspace.buffers[role]
        reused = before is not None and before.dtype == dtype and before.shape[0] >= size
        assert self.workspace.generation == generation + (not reused)
        if before is not None and not reused and self.scopes:
            self.replaced.append(before)
        self.roles.setdefault(role)

        assert (view.shape, view.dtype, held is before) == ((size,), dtype, reused)
        # A new buffer holds its capacity class: at least the size, at most half again.
        assert reused or size <= held.shape[0] <= max(1, 3 * size // 2 + 1)
        assert size == 0 or np.shares_memory(view, held)
        assert not (zeroed and view.any())
        view.fill(1)

    def __init__(self) -> None:
        super().__init__()
        self.workspace = Workspace(np)
        self.scopes: list[ExitStack] = []
        self.replaced: list[np.ndarray] = []
        self.roles: dict[str, None] = {}

    def teardown(self) -> None:
        while self.scopes:
            self.scopes.pop().close()


TestWorkspace = WorkspaceMachine.TestCase


def busy(stream: cp.cuda.Stream) -> cp.ndarray:
    """Queue on `stream` work that keeps it busy a while, then a write of ones.

    Returns the ones, an `int32` array with shape `[1 << 22]`.
    """
    with stream:
        held = cp.ones(1 << 22, np.int32)
        for _ in range(200):
            cp.cumsum(held, out=held)
        held.fill(1)
    return held


@on_gpu
def test_settle_uploads_waits_for_the_stream_it_is_given() -> None:
    stream = cp.cuda.Stream(non_blocking=True)
    busy(stream)
    settle_uploads(stream)
    assert stream.done


@on_gpu
def test_a_device_workspace_zeroes_on_the_stream_it_is_given() -> None:
    """The zeroing waits behind the busy work, so it lands after the ones."""
    stream = cp.cuda.Stream(non_blocking=True)
    workspace = Workspace(cp)
    workspace.buffers["held"] = busy(stream)
    zeroed = workspace.zeros("held", 1 << 22, np.int32, stream=stream)
    stream.synchronize()
    assert not bool(zeroed.any())
