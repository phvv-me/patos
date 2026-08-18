from collections.abc import Iterator
from contextlib import contextmanager

import torch


def eps(t: torch.Tensor) -> float:
    """Machine epsilon for `t`'s dtype, the smallest `e` with `1 + e > 1`.

    The threshold a relative precision test compares against, as in deciding whether a value
    is indistinguishable from zero at this dtype.

    t: the tensor whose dtype fixes the epsilon.
    """
    return torch.finfo(t.dtype).eps


def tiny(t: torch.Tensor) -> float:
    """Smallest positive normal value for `t`'s dtype, the safe division floor.

    `denom.clamp_min(tiny(denom))` avoids dividing by zero without distorting the non zero
    magnitudes the way a hand picked constant does.

    t: the tensor whose dtype fixes the floor.
    """
    return torch.finfo(t.dtype).tiny


def eye_like(t: torch.Tensor, n: int | None = None, *, expand: bool = True) -> torch.Tensor:
    """`torch.eye` on `t`'s device and dtype, broadcast to `t`'s batch shape by default.

    t: the tensor whose device, dtype and shape the identity matches.
    n: side length, defaulting to `t.size(-1)`, which is the common case when `t` is a square
        or thin matrix and the identity has to invert or solve against it.
    expand: when true and `t` is batched (more than two dimensions), broadcast to
        `(*t.shape[:-2], n, n)` so it lines up with a batched solve or matmul. Pass false to
        always get the bare `(n, n)` matrix.
    """
    if n is None:
        n = t.size(-1)
    out = torch.eye(n, device=t.device, dtype=t.dtype)
    if expand and t.dim() > 2:
        return out.expand(*t.shape[:-2], n, n)
    return out


@contextmanager
def fp32_matmul_precision(mode: str) -> Iterator[None]:
    """Pin the fp32 matmul precision for a region, restoring the previous mode afterwards.

    mode: `"highest"` pins true IEEE fp32 accumulation, which every exactness bearing GEMM
        needs (a nearest point certificate, a scorer against kernel equivalence pin) because
        TF32 carries only a 10 bit mantissa. `"high"` allows TF32 on Ampere and newer tensor
        cores, the speed mode for an approximate scoring GEMM whose argmax ties are
        distortion equivalent.
    """
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(mode)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
