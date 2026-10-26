"""A kernel compiles without NVVM's verifier pass, which the compile itself makes redundant."""

import compiled_code
import cupy as cp
import numpy as np
import pytest
from cuda.bindings import nvvm as bindings
from patos.cuda.backend.compiler import _nvvm
from patos.cuda.typed import Vector, cuda, kernel

_STREAM = cp.cuda.Stream.null

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")


@kernel
def stored(out: Vector[np.int32]) -> None:
    out[0] = 7


def test_a_kernel_compiles_without_the_verifier_and_nvvm_still_refuses_bad_ir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verifier checks what the compile checks again, so only the compile runs.

    A module the verifier would refuse is refused by the compile, with an NVVM error.
    """
    verified: list[int] = []
    verify = bindings.verify_program
    monkeypatch.setattr(bindings, "verify_program", lambda *a: (verified.append(1), verify(*a)))
    out = cp.zeros(1, np.int32)
    stored[1](out, stream=_STREAM)
    module = compiled_code.llvm(stored)

    major, minor = cuda.get_current_device().compute_capability
    assert (verified, out.get().tolist()) == ([], [7])
    with pytest.raises(bindings.nvvmError):
        _nvvm([module.replace("ret void", "ret i32 0", 1)], {"arch": f"compute_{major}{minor}"})
