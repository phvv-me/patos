import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


def seed_all(seed: int = 0) -> None:
    """Seed python, numpy and torch (CPU and every CUDA device) from one call.

    The single RNG entry point for a process, so library code draws from the default global
    generator and never threads one of its own. The deliberate exception is a reproducible
    multi restart search, which seeds an explicit generator per restart on purpose.

    seed: the value every generator starts from.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Draw inside the block from `seed`, handing the caller's RNG back untouched afterwards.

    The scoped counterpart to `seed_all`, for a draw that must come out identical on every run
    and every device (a sign vector, a rotation, a synthetic source) without advancing the
    global stream that everything after it reads. Only the CPU generator is forked, since a
    reproducible draw is made on CPU and moved to the device rather than drawn there.

    seed: the value the forked CPU generator starts from.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def configure_torch(*, matmul_precision: str = "high") -> None:
    """Enable the TF32 fast paths and set float32 matmul precision, once per process.

    matmul_precision: torch float32 matmul precision, where `"high"` uses TF32 on Ampere
        and newer.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision(matmul_precision)


def setup(seed: int = 0, *, matmul_precision: str = "high") -> None:
    """Seed everything and configure torch precision in one call at process entry.

    seed: the value every generator starts from.
    matmul_precision: torch float32 matmul precision.
    """
    seed_all(seed)
    configure_torch(matmul_precision=matmul_precision)
