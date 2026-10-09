"""The numeric types import where no CUDA stack is installed, as host table builders need."""

import json
import subprocess
import sys

# A fresh interpreter in which numba, numba-cuda, CuPy, cuda-python and llvmlite cannot import.
_WITHOUT_CUDA = """
import json, sys

stack = {"numba", "numba_cuda", "cupy", "cuda", "llvmlite"}
for name in (*stack, "numba.cuda", "cuda.core"):
    sys.modules[name] = None

import numpy as np

from patos.cuda.scalars import SPELLINGS, ArrayOf, Subscript, i16, u8, u8x4, u32, u64

print(json.dumps({
    "array": u8[int, int] == ArrayOf(np.uint8, 2),
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
