"""Where compiled units and fetched header libraries are kept between processes."""

import atexit
import os
import shutil
import tempfile
import warnings
from functools import cache
from pathlib import Path


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
