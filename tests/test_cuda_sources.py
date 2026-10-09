"""A kernel's sources are the first-party modules its module reaches, whichever others a process
imported, so the folder they name moves with an edit to one of them and with nothing else."""

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

_MAIN = """
import sys
from importlib import import_module

import layout
from helper import bump
from patos.cuda.sources import digest, reached

for unrelated in sys.argv[1:]:
    import_module(unrelated)


def run(out):
    out[0] = bump(layout.Tile.width)


print(digest(reached(sys.modules[run.__module__]), seed="toolchain"))
"""
_PROJECT = {
    "main.py": _MAIN,
    "helper.py": "def bump(value):\n    return value + 10\n",
    "layout/__init__.py": "from .tile import Tile\n",
    "layout/tile.py": "class Tile:\n    width = 4\n",
    "noise.py": "NOISE = 1\n",
    "tools/__init__.py": "",
    "tools/spare.py": "SPARE = 1\n",
}
_UNRELATED = ("noise", "tools.spare")
_UNRELATED_EDITS = {"noise.py": "NOISE = 2\n", "tools/spare.py": "SPARE = 2\n"}
_REACHED_EDITS = {
    "helper.py": "def bump(value):\n    return value + 20\n",
    "layout/tile.py": "class Tile:\n    width = 8\n",
}


def written(project: Path, files: Mapping[str, str]) -> Path:
    """`project`, with each file of `files` written to its text."""
    for name, text in files.items():
        (project / name).parent.mkdir(exist_ok=True)
        (project / name).write_text(text)
    return project


def named(project: Path, *unrelated: str) -> str:
    """The digest `project/main.py` names its kernel's folder by, in a process of its own.

    unrelated: the modules that process imports too.
    """
    environment = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(project), *sys.path]),
    }
    done = subprocess.run(
        [sys.executable, "main.py", *unrelated],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return done.stdout.strip()


def test_a_kernel_s_folder_follows_what_its_module_reaches_and_nothing_else(
    tmp_path: Path,
) -> None:
    """Modules the kernel never reaches, imported or edited, leave its folder where it was.

    The kernel's module is `__main__`, and the second process also imports a module and a
    package's submodule it never reaches. An edit to the helper it calls, or to the class the
    package it binds holds, moves the folder.
    """
    project = written(tmp_path, _PROJECT)
    plain, crowded = named(project), named(project, *_UNRELATED)
    untouched = named(written(project, _UNRELATED_EDITS), *_UNRELATED)
    moved = [named(written(project, {name: text})) for name, text in _REACHED_EDITS.items()]

    assert plain == crowded == untouched
    assert len({plain, *moved}) == 3
