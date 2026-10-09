"""The first-party sources a module's code can reach, which import without the CUDA stack.

A module reaches the modules its globals are, and the modules that define what its globals hold:
a class, a function, a record type or a device function names its own. Followed through every
first-party module, a Python file outside the environment's packages, these are the sources a
kernel the module defines can call, whichever other modules the process imported. Only a package
reached as a global carries more: the import system sets every submodule imported anywhere on it.
A function-local import goes unseen, which the house rule forbids.
"""

import hashlib
import sys
import sysconfig
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from types import ModuleType

_INSTALLED = (*(sysconfig.get_path(name) for name in ("stdlib", "platstdlib", "purelib")),)


@cache
def reached(module: ModuleType) -> tuple[str, ...]:
    """The files of the first-party modules `module` reaches, its own among them, sorted.

    Cached per module: a kernel's module has bound every global by the time its first kernel
    compiles.
    """
    seen, pending = {module}, [module]
    while pending:
        for found in _held(pending.pop()):
            if found not in seen and _source(found):
                seen.add(found)
                pending.append(found)
    return (*sorted(filter(None, map(_source, seen))),)


def digest(files: Iterable[str], *, seed: str) -> str:
    """The digest of `seed` and of each file's path and bytes.

    seed: what else the digest covers, such as the toolchain that compiles the sources.
    """
    combined = hashlib.sha256(seed.encode())
    for path in sorted(files):
        combined.update(f"\0{path}\0".encode() + _content(path))
    return combined.hexdigest()[:32]


def _held(holder: ModuleType) -> Iterator[ModuleType]:
    """The modules `holder`'s globals are or were defined in."""
    for value in list(vars(holder).values()):
        if not isinstance(value, ModuleType):
            value = sys.modules.get(getattr(value, "__module__", None) or "")
        if value is not None:
            yield value


def _source(module: ModuleType) -> str | None:
    """The file of `module` when it is first-party."""
    file = getattr(module, "__file__", None) or ""
    return file if file.endswith(".py") and not file.startswith(_INSTALLED) else None


@cache
def _content(path: str) -> bytes:
    """The digest of the bytes at `path`, read once; empty for a file deleted since its import."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).digest()
    except FileNotFoundError:
        return b""
