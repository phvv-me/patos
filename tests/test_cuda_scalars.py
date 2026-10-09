"""The numeric types import where no CUDA stack is installed, as host table builders need."""

import json
import operator
import re
import subprocess
import sys

import numpy as np
import pytest

from patos.cuda import scalars

# A fresh interpreter in which numba, numba-cuda, CuPy, cuda-python and llvmlite cannot import.
_WITHOUT_CUDA = """
import json, sys

stack = {"numba", "numba_cuda", "cupy", "cuda", "llvmlite"}
for name in (*stack, "numba.cuda", "cuda.core"):
    sys.modules[name] = None

import numpy as np

from patos.cuda.scalars import SPELLINGS, ArrayOf, Matrix, Subscript, i16, u8, u8x4, u32, u64

print(json.dumps({
    "array": Matrix[u8] == ArrayOf(element=np.uint8, ndim=2),
    "scalar": type(u32(7)).__name__,
    "spelled": SPELLINGS[np.int16],
    "dtype": np.dtype(i16).name,
    "lanes": u8x4.name,
    "subscript": Subscript.__name__,
    "wide": int(u64(2**64 - 1)),
    "loaded": [name for name, held in sys.modules.items() if held and name.split(".")[0] in stack],
}))
"""


def test_the_numeric_types_import_without_numba_cupy_or_cuda() -> None:
    """`patos.cuda.scalars` loads no CUDA module, and its types annotate and convert as ever."""
    ran = subprocess.run(
        [sys.executable, "-c", _WITHOUT_CUDA], capture_output=True, text=True, timeout=120
    )

    assert ran.returncode == 0, ran.stderr
    assert json.loads(ran.stdout) == {
        "array": True,
        "scalar": "uint32",
        "spelled": "i16",
        "dtype": "int16",
        "lanes": "u8x4",
        "subscript": "Subscript",
        "wide": 2**64 - 1,
        "loaded": [],
    }


def test_vector_and_matrix_declare_arrays_of_a_numeric_type() -> None:
    assert scalars.Vector[scalars.u8] == scalars.ArrayOf(element=np.uint8, ndim=1)
    assert scalars.Matrix[scalars.i16] == scalars.ArrayOf(element=np.int16, ndim=2)
    assert scalars.Vector[scalars.number] == scalars.ArrayOf(element=np.number, ndim=1)
    assert scalars.Vector[scalars.f32] == scalars.ArrayOf(element=np.float32, ndim=1)
    with pytest.raises(TypeError, match="holds a numeric type"):
        operator.getitem(scalars.Vector, int)


@pytest.mark.parametrize(
    ("kind", "shape", "written"),
    [(scalars.u8, int, "Vector[u8]"), (scalars.i16, (int, int), "Matrix[i16]")],
)
def test_the_placeholder_spelling_of_an_array_is_refused(
    kind: type[np.number], shape: type[int] | tuple[type[int], ...], written: str
) -> None:
    with pytest.raises(TypeError, match=re.escape(f"write {written}")):
        kind.__class_getitem__(shape)
