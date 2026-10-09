"""The folder a kernel is cached in: named by the digest of the sources its code can reach.

numba-cuda's disk cache keeps a kernel under its own source file and notices a change to that
file alone, so a helper module edited or a toolkit upgraded would load a stale kernel. The folder
is named by the digest of the first-party sources the kernel's module reaches
(`patos.cuda.sources`), every module of patos's compiler, the toolchain's versions and the GPU. An
edit anywhere the kernel's code can reach moves its folder, so no kernel loads stale, and a
process that imported other modules finds the same folder. Folders unused for two weeks are
deleted when a new one is made.
"""

import importlib.metadata
import os
import shutil
import sys
import time
from contextlib import suppress
from functools import cache
from pathlib import Path
from types import FunctionType, ModuleType

from numba import cuda
from numba.cuda.core.caching import _CacheLocator, _SourceFileBackedLocatorMixin

from ..sources import digest, reached
from .cachedir import user_cache

# The distributions whose versions decide the code a kernel compiles to.
_TOOLCHAIN = (
    "numba", "numba-cuda", "cuda-core", "cuda-bindings", "nvidia-nvvm", "nvidia-nvjitlink",
    "nvidia-cuda-nvcc", "nvidia-cuda-nvrtc",
)  # fmt: skip
# patos's compiler, whose every module decides the code a kernel compiles to.
_COMPILER = (*map(str, Path(__file__).parent.glob("*.py")),)
_UNUSED_SECONDS = 14 * 24 * 3600


class Locator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """Places a kernel's files in the folder of the sources its module reaches, by the folder of
    its file."""

    def __init__(self, py_func: FunctionType, py_file: str) -> None:
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        folder = _made(user_cache() / "kernels", _named(sys.modules[py_func.__module__]))
        self._cache_path = str(folder / self.get_suitable_cache_subpath(py_file))

    def get_cache_path(self) -> str:
        return self._cache_path


@cache
def _named(module: ModuleType) -> str:
    """The name of the folder of `module`'s kernels."""
    return digest({*reached(module), *_COMPILER}, seed=_toolchain())


@cache
def _made(root: Path, name: str) -> Path:
    """The folder `name` in `root`, touched, with the folders unused for long deleted."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    os.utime(folder)
    for unused in root.iterdir():
        if time.time() - unused.stat().st_mtime > _UNUSED_SECONDS:
            shutil.rmtree(unused, ignore_errors=True)
    return folder


@cache
def _toolchain() -> str:
    """Python, the packages that compile a kernel, and the GPU's compute capability."""
    versions = []
    for name in _TOOLCHAIN:
        with suppress(importlib.metadata.PackageNotFoundError):
            versions.append(f"{name} {importlib.metadata.version(name)}")
    return "\n".join([sys.version, *versions, str(cuda.get_current_device().compute_capability)])
