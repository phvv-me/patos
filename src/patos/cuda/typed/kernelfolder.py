"""The folder a kernel is cached in: named by the digest of the sources it is made of.

numba-cuda's disk cache keeps a kernel under its own source file and notices a change to that
file alone, so a helper module edited or a toolkit upgraded would load a stale kernel. The folder
is named by the digest of every first-party source imported, the toolchain's versions and the GPU:
an edit anywhere moves the folder, and no kernel loads stale. Folders unused for two weeks are
deleted when a new one is made.
"""

import hashlib
import importlib.metadata
import os
import shutil
import sys
import sysconfig
import time
from collections.abc import Sequence
from contextlib import suppress
from functools import cache
from pathlib import Path
from types import FunctionType

from numba import cuda
from numba.cuda.core.caching import _CacheLocator, _SourceFileBackedLocatorMixin

from .cachedir import user_cache

# The distributions whose versions decide the code a kernel compiles to.
_TOOLCHAIN = (
    "numba", "numba-cuda", "cuda-core", "cuda-bindings", "nvidia-nvvm", "nvidia-nvjitlink",
    "nvidia-cuda-nvcc", "nvidia-cuda-nvrtc",
)  # fmt: skip
_UNUSED_SECONDS = 14 * 24 * 3600


class Locator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """Places a kernel's files in the folder of the sources imported, by the folder of its file."""

    def __init__(self, py_func: FunctionType, py_file: str) -> None:
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        folder = _made(user_cache() / "kernels", _digest(self._sources()))
        self._cache_path = str(folder / self.get_suitable_cache_subpath(py_file))

    def get_cache_path(self) -> str:
        return self._cache_path

    @staticmethod
    def _sources() -> tuple[str, ...]:
        """The Python files of every module imported from outside the environment's packages.

        Plain strings, as a kernel asks for this each time it is cached: it takes 3 ms, not 60.
        """
        installed = (*(sysconfig.get_path(n) for n in ("stdlib", "platstdlib", "purelib")),)
        files = {str(getattr(m, "__file__", None) or "") for m in list(sys.modules.values())}
        return (*sorted(f for f in files if f.endswith(".py") and not f.startswith(installed)),)


@cache
def _made(root: Path, digest: str) -> Path:
    """The folder `digest` names in `root`, touched, with the folders unused for long deleted."""
    folder = root / digest
    folder.mkdir(parents=True, exist_ok=True)
    os.utime(folder)
    for unused in root.iterdir():
        if time.time() - unused.stat().st_mtime > _UNUSED_SECONDS:
            shutil.rmtree(unused, ignore_errors=True)
    return folder


@cache
def _digest(sources: Sequence[str]) -> str:
    """The digest of the files `sources` name, the toolchain's versions and the GPU."""
    digest = hashlib.sha256(_toolchain().encode())
    for path in sources:
        with suppress(FileNotFoundError):
            digest.update(f"\0{path}\0".encode() + Path(path).read_bytes())
    return digest.hexdigest()[:32]


@cache
def _toolchain() -> str:
    """Python, the packages that compile a kernel, and the GPU's compute capability."""
    versions = []
    for name in _TOOLCHAIN:
        with suppress(importlib.metadata.PackageNotFoundError):
            versions.append(f"{name} {importlib.metadata.version(name)}")
    return "\n".join([sys.version, *versions, str(cuda.get_current_device().compute_capability)])
