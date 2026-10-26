"""Kernels compiled once for the sources they are made of: a cold process compiles and keeps
their cubins, a warm one loads them, and a source changed anywhere compiles them again."""

import json
import os
import subprocess
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from patos.cuda.typed import Struct, Vector, i32, kernel

_STREAM = cp.cuda.Stream.null

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

_HELPER = """
from patos.cuda.typed import device, i32

@device
def bump(x: i32) -> i32:
    return x + {offset}
"""
# Counts the kernels the process compiled rather than loaded, which the backend logs.
_COUNTED = """
import json
import logging

compiled = []
counting = logging.getLogger("patos.cuda.backend.kernels")
counting.setLevel(logging.INFO)
counting.addHandler(logging.Handler())
counting.handlers[-1].emit = lambda record: compiled.append(record.getMessage())
"""
# The child writes what it ran to the file `REPORT` names.
_REPORT = """
import os
from pathlib import Path

Path(os.environ["REPORT"]).write_text(json.dumps({"result": result, "compiled": len(compiled)}))
"""
_PLAIN = """
import cupy as cp
import numpy as np
from helper import bump
from patos.cuda.typed import Vector, i32, kernel

@kernel
def run(out: Vector[i32]) -> None:
    out[0] = bump(1)

out = cp.zeros(1, np.int32)
run[1](out, stream=cp.cuda.Stream.null)
result = int(out[0])
"""
_LINKED = """
import cupy as cp
import numpy as np
from patos.cuda.typed import Cxx, Vector, cccl, i32, kernel, thread_index

cub = Cxx(cccl, "#include <cub/warp/warp_reduce.cuh>", name="cub")

@cub('''
    using Reduce = cub::WarpReduce<cuda::std::int32_t>;
    __shared__ Reduce::TempStorage storage[4];
    return Reduce(storage[threadIdx.x / 32]).Sum(value);
''')
def total(value: i32) -> i32:
    raise NotImplementedError

@kernel(threads=128)
def run(values: Vector[i32], out: Vector[i32]) -> None:
    index = thread_index()
    summed = total(values[index])
    if index % 32 == 0:
        out[index // 32] = summed

out = cp.zeros(4, np.int32)
run[128](cp.arange(128, dtype=np.int32), out, stream=cp.cuda.Stream.null)
result = out.get().tolist()
"""


_MISMATCHED = """
import cupy as cp
import numpy as np
from patos.cuda.primitives import block
from patos.cuda.typed import Vector, i32, kernel, thread_index

_ACROSS = block.over(128)

@kernel(threads=64)
def run(values: Vector[i32], out: Vector[i32]) -> None:
    out[thread_index()] = _ACROSS.sum(values[thread_index()])

try:
    run[64](cp.ones(64, np.int32), cp.zeros(64, np.int32), stream=cp.cuda.Stream.null)
    result = "ran"
except TypeError as error:
    result = str(error)
"""


def launched(project: Path) -> dict[str, int | str | list[int]]:
    """What `project/main.py` reports when a new process runs it, its cache beside the project."""
    environment = os.environ | {
        "XDG_CACHE_HOME": str(project.parent / "cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(project), *sys.path]),
        "REPORT": str(project.parent / "report.json"),
    }
    subprocess.run(
        [sys.executable, "main.py"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    return json.loads((project.parent / "report.json").read_text())


def test_a_kernel_compiles_once_and_a_changed_helper_compiles_it_again(tmp_path: Path) -> None:
    """Cold compiles and keeps the kernel, and warm loads it.

    An edited helper module, which numba's own cache never sees, moves the folder, so no stale
    kernel loads.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(_COUNTED + _PLAIN + _REPORT)
    outcomes = []
    for offset in (10, 10, 20, 20):
        (project / "helper.py").write_text(_HELPER.format(offset=offset))
        outcomes.append(launched(project))

    assert outcomes == [
        {"result": 11, "compiled": 1},
        {"result": 11, "compiled": 0},
        {"result": 21, "compiled": 1},
        {"result": 21, "compiled": 0},
    ]


def test_a_kernel_linking_a_cxx_unit_is_kept_once_it_is_linked(tmp_path: Path) -> None:
    """The kernel's cubin is the linked result, so a warm process needs no unit to link."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(_COUNTED + _LINKED + _REPORT)
    cold, warm = (launched(project) for _ in range(2))

    sums = [sum(range(32 * warp, 32 * warp + 32)) for warp in range(4)]
    assert (cold, warm) == ({"result": sums, "compiled": 1}, {"result": sums, "compiled": 0})


def test_a_kernel_of_a_record_made_in_a_function_compiles_and_is_not_kept() -> None:
    """Another process cannot find the class again, so the kernel is built in each process."""

    class Local(Struct):
        slots: Vector[i32]

    @kernel
    def write(record: Local) -> None:
        record.slots[0] = 5

    slots = cp.zeros(1, np.int32)
    write[1](Local(slots), stream=_STREAM)

    assert slots.get().tolist() == [5]


def test_a_kernel_refused_at_its_launch_is_not_kept_for_a_process_that_would_not_check_it(
    tmp_path: Path,
) -> None:
    """A block reduction made for 128 threads in a kernel of 64 is refused at its compile.

    A kernel kept anyway would load, in the next process, with that check skipped.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(_COUNTED + _MISMATCHED + _REPORT)
    outcomes = [launched(project) for _ in range(2)]

    assert [("made for 128" in str(found["result"]), found["compiled"]) for found in outcomes] == [
        (True, 1),
        (True, 1),
    ]
