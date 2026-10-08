"""CUDA device code in plain, annotated Python on numba-cuda, and the host runtime that launches
it.

Installed with the `cuda` extra. `typed` holds the annotation typing and the `device` and `kernel`
decorators, `primitives` the warp folds and hash tables kernels share, and `runtime` the launch,
stream and scratch-memory plumbing a kernel library repeats.
"""
