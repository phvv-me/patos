from contextlib import ExitStack
from itertools import combinations
from types import SimpleNamespace

import numpy as np
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from patos.cuda.runtime import Workspace, settle_uploads

roles = st.sampled_from(["scan", "keys", "out"])
dtypes = st.sampled_from([np.uint8, np.int32, np.float64])


class WorkspaceMachine(RuleBasedStateMachine):
    """Takes, zeroed takes and nested retention scopes in any order.

    A take is an exact view of its role's buffer, replaced only to grow or to change dtype; no two
    roles share storage; a zeroed take clears what an earlier one left; and every buffer replaced
    inside a scope stays alive until the outermost scope closes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.workspace = Workspace(np)
        self.scopes: list[ExitStack] = []
        self.replaced: list[np.ndarray] = []
        self.roles: dict[str, None] = {}

    @rule()
    def enter(self) -> None:
        scope = ExitStack()
        scope.enter_context(self.workspace.retain_replaced())
        self.scopes.append(scope)

    @invariant()
    def keeps_each_role_apart(self) -> None:
        buffers = self.workspace.buffers.values()
        assert list(self.workspace) == list(self.roles)
        assert not any(
            np.shares_memory(first, second) for first, second in combinations(buffers, 2)
        )

    @precondition(lambda machine: machine.scopes)
    @rule()
    def leave(self) -> None:
        self.scopes.pop().close()
        if not self.scopes:
            self.replaced.clear()

    @invariant()
    def retains_what_a_scope_replaced(self) -> None:
        assert [id(buffer) for buffer in self.workspace.retained] == list(map(id, self.replaced))
        assert self.workspace.retention_depth == len(self.scopes)

    @rule(role=roles, size=st.integers(0, 40), dtype=dtypes, zeroed=st.booleans())
    def take(self, role: str, size: int, dtype: type[np.generic], *, zeroed: bool) -> None:
        before = self.workspace.buffers.get(role)
        generation = self.workspace.generation
        view = (self.workspace.zeros if zeroed else self.workspace.take)(role, size, dtype)
        held = self.workspace.buffers[role]
        reused = before is not None and before.dtype == dtype and before.shape[0] >= size
        assert self.workspace.generation == generation + (not reused)
        if before is not None and not reused and self.scopes:
            self.replaced.append(before)
        self.roles.setdefault(role)

        assert (view.shape, view.dtype, held is before) == ((size,), dtype, reused)
        assert reused or held.shape[0] == max(size, 1)
        assert size == 0 or np.shares_memory(view, held)
        assert not (zeroed and view.any())
        view.fill(1)

    def teardown(self) -> None:
        while self.scopes:
            self.scopes.pop().close()


TestWorkspace = WorkspaceMachine.TestCase


def test_settle_uploads_synchronizes_the_current_stream_and_skips_modules_without_one() -> None:
    """An array module with no CUDA streams, such as numpy, has nothing to settle."""
    synchronized: list[None] = []
    stream = SimpleNamespace(synchronize=lambda: synchronized.append(None))

    settle_uploads(SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda: stream)))
    settle_uploads(np)

    assert len(synchronized) == 1
