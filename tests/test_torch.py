import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from patos.torch import (
    configure_torch,
    eps,
    eye_like,
    fp32_matmul_precision,
    seed_all,
    seeded,
    setup,
    tiny,
)

seeds = st.integers(min_value=0, max_value=2**31 - 1)


@pytest.fixture(autouse=True, scope="module")
def restore_torch_globals() -> Iterator[None]:
    """Hand the process-wide torch switches these tests flip back to the runner afterwards."""
    precision = torch.get_float32_matmul_precision()
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.benchmark = benchmark


@given(seed=seeds)
@settings(deadline=None)
def test_seed_all_replays_python_numpy_and_torch_from_one_call(seed: int) -> None:
    """One call fixes all three streams, so the same seed replays the same draws."""
    seed_all(seed)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_all(seed)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


@given(seed=seeds)
@settings(deadline=None)
def test_seeded_replays_its_block_and_leaves_the_caller_stream_untouched(seed: int) -> None:
    """The block is reproducible while the surrounding generator resumes where it stopped."""
    torch.manual_seed(0)
    expected = torch.rand(4)

    torch.manual_seed(0)
    with seeded(seed):
        inside = torch.rand(4)
    with seeded(seed):
        assert torch.equal(torch.rand(4), inside)
    assert torch.equal(torch.rand(4), expected)


def test_configure_torch_pins_the_requested_matmul_precision() -> None:
    """The precision argument reaches torch and the cudnn autotuner is switched on."""
    configure_torch(matmul_precision="highest")

    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cudnn.benchmark is True


def test_setup_seeds_and_configures_in_one_call() -> None:
    """setup is seed_all and configure_torch together, the one line a process entry runs."""
    setup(7)
    drawn = torch.rand(2)

    assert torch.get_float32_matmul_precision() == "high"
    seed_all(7)
    assert torch.equal(torch.rand(2), drawn)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_eps_and_tiny_read_the_dtype_of_the_tensor(dtype: torch.dtype) -> None:
    """Both follow `t`'s dtype rather than a hard coded constant, and tiny floors below eps."""
    t = torch.zeros(3, dtype=dtype)

    assert eps(t) == torch.finfo(dtype).eps
    assert tiny(t) == torch.finfo(dtype).tiny
    assert 0 < tiny(t) < eps(t)


@given(n=st.integers(min_value=1, max_value=8))
@settings(deadline=None)
def test_eye_like_is_an_identity_carrying_the_tensor_dtype_and_device(n: int) -> None:
    """The result matches `t`'s dtype and device and leaves `t` unchanged under matmul."""
    t = torch.randn(n, n, dtype=torch.float64)
    identity = eye_like(t)

    assert identity.shape == (n, n)
    assert identity.dtype == t.dtype
    assert identity.device == t.device
    assert torch.equal(t @ identity, t)


def test_eye_like_takes_an_explicit_side_and_broadcasts_over_batch_dims() -> None:
    """`n` overrides the trailing size and a batched `t` expands unless `expand` is off."""
    batched = torch.zeros(2, 5, 3, 4)

    assert eye_like(batched).shape == (2, 5, 4, 4)
    assert eye_like(batched, expand=False).shape == (4, 4)
    assert eye_like(batched, 2).shape == (2, 5, 2, 2)
    assert eye_like(torch.zeros(3, 4)).shape == (4, 4)


def test_fp32_matmul_precision_pins_a_region_and_restores_on_the_way_out() -> None:
    """The mode holds inside the block and the previous mode comes back, exception or not."""
    torch.set_float32_matmul_precision("high")

    with fp32_matmul_precision("highest"):
        assert torch.get_float32_matmul_precision() == "highest"
    assert torch.get_float32_matmul_precision() == "high"

    with pytest.raises(RuntimeError, match="boom"), fp32_matmul_precision("highest"):
        raise RuntimeError("boom")
    assert torch.get_float32_matmul_precision() == "high"
