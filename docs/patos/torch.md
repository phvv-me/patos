# Torch

`patos.torch` is the optional torch namespace. It holds the run-wide RNG and precision controls
plus the four dtype-aware tensor helpers every torch codebase ends up rewriting, and it keeps
torch and numpy out of the core Patos installation.

## Install

```sh
pip install "patos[torch]"
```

The torch floor is loose (`torch>=2`), since nothing here reaches past `manual_seed`, `finfo`,
`eye` and the matmul precision switches. Your project keeps whatever exact build its accelerator
lane needs.

```python
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
```

## One RNG entry point

`seed_all` seeds python, numpy and torch (CPU and every CUDA device) together, and
`configure_torch` turns on the TF32 fast paths and pins float32 matmul precision. `setup` is the
single line a process entry runs to do both.

```python
setup(seed=0, matmul_precision="high")
```

Library code below that entry point never seeds. It draws from the default global generator, so
one call at the top of a run fixes the whole process.

## Scoped, reproducible draws

`seeded` is the scoped counterpart, for a draw that has to come out identical on every run and
every device without advancing the global stream that everything after it reads.

```python
with seeded(1234):
    signs = torch.randint(0, 2, (dim,)) * 2 - 1
```

Only the CPU generator is forked, because a reproducible draw is made on CPU and moved to the
device rather than drawn there.

## Dtype-aware tensor helpers

`eps` and `tiny` read the tensor's own dtype instead of a hard-coded constant, which is what makes
the same code correct in float64, float32 and bfloat16.

```python
ratio = numerator / denominator.clamp_min(tiny(denominator))
converged = bool(step.abs().max() < eps(step))
```

`eye_like` builds `torch.eye` on the tensor's device and dtype, broadcast over its batch
dimensions so it lines up with a batched `solve` or `matmul`.

```python
inverse = torch.linalg.solve(batched, eye_like(batched))
raw = eye_like(batched, expand=False)  # the bare (n, n) matrix
```

`fp32_matmul_precision` pins the precision for one region and restores the previous mode on the
way out, including when the block raises. Use `"highest"` wherever a GEMM is exactness-bearing,
since TF32 carries only a 10-bit mantissa.

```python
with fp32_matmul_precision("highest"):
    certificate = queries @ codebook.T
```
