"""Compiled code kept between processes in the user's cache directory, each unit under the digest
of everything that decides its bytes.

A kernel's cubin is keyed by the first-party sources its module reaches (`sources`), every source
of patos.cuda, the versions of the packages that compile it, Python's and the GPU's compute
capability, so an edit anywhere its code can reach, or an upgrade, compiles it again, and a process
that imported other modules finds it all the same. It loads through cuda.core's `ObjectCode`, so
no compiler is involved in a process that finds it. Kernels unused for two weeks are deleted when
a new one is kept. Where the cache directory cannot be written, a temporary folder stands in, and
each process compiles.
"""

import atexit
import importlib.metadata
import os
import shutil
import sys
import tempfile
import time
import warnings
from contextlib import suppress
from functools import cache
from pathlib import Path
from types import ModuleType

from cuda.core import Device, ObjectCode
from cuda.core import Kernel as Loaded

from ..bases import FrozenModel, Runtime
from .sources import digest, reached

# The header libraries C++ units are pinned to.
_LIBRARIES = Path(__file__).with_name("backend") / "libraries.toml"
# The distributions whose versions decide the code a kernel compiles to.
_TOOLCHAIN = (
    "patos", "numba", "numba-cuda", "llvmlite", "cuda-core", "cuda-bindings", "cuda-cccl",
    "cuda-toolkit", "nvidia-nvvm", "nvidia-nvjitlink", "nvidia-cuda-nvcc", "nvidia-cuda-nvrtc",
)  # fmt: skip
# Every source of patos.cuda, whose compiler, device primitives and pinned libraries decide the
# code a kernel compiles to, whether patos is installed from a wheel or from its sources.
_PATOS = (*sorted(str(path) for path in Path(__file__).parent.rglob("*.py")), str(_LIBRARIES))
_UNUSED_SECONDS = 14 * 24 * 3600


class Cubin(FrozenModel):
    """A kernel's linked code and the name of its entry point in it."""

    code: Runtime[ObjectCode]
    entry: str

    def kernel(self) -> Loaded:
        return self.code.get_kernel(self.entry)


def user_cache() -> Path:
    """patos's folder in the user's cache directory, or a temporary one where that has no room.

    Nothing then outlives the process: a unit compiles, and a library fetches, in each one.
    """
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache", "patos")
    try:
        _made_writable(root)
    except OSError:
        return _uncached(root)
    return root


def stored(path: Path, data: bytes) -> None:
    """Write `data` to `path` whole: another process reads either nothing or every byte."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as partial:
        partial.write(data)
    Path(partial.name).replace(path)


@cache
def sources_digest(module: ModuleType) -> str:
    """The digest of every source a kernel `module` defines can reach, and of what compiles it."""
    return digest({*reached(module), *_PATOS}, seed=_toolchain())


def loaded(key: str) -> Cubin | None:
    """The cubin kept under `key`, None when no process kept one."""
    path = _folder() / f"{key}.cubin"
    try:
        entry, _, cubin = path.read_bytes().partition(b"\0")
    except FileNotFoundError:
        return None
    with suppress(OSError):
        os.utime(path)
    return Cubin(code=ObjectCode.from_cubin(cubin), entry=entry.decode())


def kept(key: str, cubin: Cubin) -> None:
    """Keep `cubin` under `key`, deleting the kernels unused for two weeks first."""
    _pruned()
    stored(_folder() / f"{key}.cubin", cubin.entry.encode() + b"\0" + bytes(cubin.code.code))


@cache
def _folder() -> Path:
    folder = user_cache() / "cubins"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@cache
def _pruned() -> None:
    """Delete the kept kernels no process loaded for two weeks, once a process."""
    for unused in _folder().iterdir():
        with suppress(OSError):
            if time.time() - unused.stat().st_mtime > _UNUSED_SECONDS:
                unused.unlink()


@cache
def _toolchain() -> str:
    """Python, the packages that compile a kernel, and the GPU's compute capability."""
    versions = []
    for name in _TOOLCHAIN:
        with suppress(importlib.metadata.PackageNotFoundError):
            versions.append(f"{name} {importlib.metadata.version(name)}")
    return "\n".join([sys.version, *versions, str(tuple(Device().compute_capability))])


def _made_writable(root: Path) -> None:
    """Make `root` and prove a file can be made in it."""
    root.mkdir(parents=True, exist_ok=True)
    tempfile.TemporaryFile(dir=root).close()


@cache
def _uncached(unwritable: Path) -> Path:
    """A temporary folder in place of `unwritable`, removed at exit; the first use says so."""
    message = f"{unwritable} cannot be written; compiling without caching"
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    folder = Path(tempfile.mkdtemp(prefix="patos-"))
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    return folder
