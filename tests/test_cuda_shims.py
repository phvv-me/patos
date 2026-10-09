"""The patches of numba-cuda that patos applies."""

import cupy as cp
import numpy as np
import pytest
from cuda.bindings import nvvm as bindings
from numba.cuda.cudadrv import nvvm

from patos.cuda.typed import Vector, cuda, kernel

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")


@kernel
def stored(out: Vector[np.int32]) -> None:
    out[0] = 7


def test_nvvm_compiles_a_kernel_without_the_verifier_pass_and_still_refuses_bad_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verifier checks what the compile checks again, so only the compile runs.

    A module the verifier would refuse is refused by the compile, with the same error.
    """
    verified: list[int] = []
    verify = bindings.verify_program
    monkeypatch.setattr(bindings, "verify_program", lambda *a: (verified.append(1), verify(*a)))
    out = cp.zeros(1, np.int32)
    stored[1](out)

    major, minor = cuda.get_current_device().compute_capability
    module = stored.dispatcher.inspect_llvm(next(iter(stored.dispatcher.overloads)))
    assert (verified, out.get().tolist()) == ([], [7])
    with pytest.raises(bindings.nvvmError):
        nvvm.compile_ir(module.replace("ret void", "ret i32 0", 1), arch=f"compute_{major}{minor}")
