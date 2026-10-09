"""CUDA device code in plain, annotated Python on numba-cuda, and the host runtime that launches
it.

Installed with the `cuda` extra. `typed` holds the annotation typing and the `device` and `kernel`
decorators, `primitives` the warp folds and hash tables kernels share, `runtime` the launch,
stream and scratch-memory plumbing a kernel library repeats, and `graphs` the passes recorded
once and replayed for the cost of a launch.
"""
