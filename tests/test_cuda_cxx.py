"""C++ from a header library behind an annotated stub: CUB's warp sum, inlined and cached."""

import logging
import re
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

from patos.cuda.primitives import warp
from patos.cuda.typed import (
    Compiler,
    Cxx,
    Headers,
    Kernel,
    Vector,
    cccl,
    i32,
    kernel,
    thread_index,
)

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")

_REDUCE = """
    using Reduce = cub::WarpReduce<cuda::std::int32_t>;
    __shared__ Reduce::TempStorage storage[4];
    return Reduce(storage[threadIdx.x / 32]).Sum(value);
"""


def unit(
    compiler: Compiler = Compiler.NVRTC, *, defines: Sequence[str] = ("NDEBUG",)
) -> tuple[Cxx, Kernel]:
    """A unit holding CUB's warp sum, and a kernel writing each warp's sum from its lane 0.

    Each call makes a unit of its own, which remembers no compile, so it reads the cache.
    """
    headers = partial(compiled_by, compiler, list(defines))
    cub = Cxx(headers, "#include <cub/warp/warp_reduce.cuh>", name="cub")

    @cub(_REDUCE)
    def warp_total(value: i32) -> i32:
        """The sum of `value` across the warp, in lane 0."""
        raise NotImplementedError

    @kernel(threads=128)
    def sums(values: Vector[i32], out: Vector[i32]) -> None:
        index = thread_index()
        total = warp_total(values[index])
        if index % 32 == 0:
            out[index // 32] = total

    return cub, sums


def compiled_by(compiler: Compiler, defines: list[str]) -> Headers:
    """CCCL, compiled by `compiler` with the macros `defines`."""
    return cccl().model_copy(update={"compiler": compiler, "defines": defines})


@kernel(threads=128)
def patos_sums(values: Vector[i32], out: Vector[i32]) -> None:
    index = thread_index()
    total = warp.sum(values[index])
    if index % 32 == 0:
        out[index // 32] = total


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty user cache, where the units compiled in a test are kept."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path / "patos" / "cxx"


def summed(launched: Kernel) -> list[int]:
    """What `launched` makes of 4096 random words: one sum per warp."""
    values = np.random.default_rng(3).integers(-1000, 1000, 4096, dtype=np.int32)
    out = cp.zeros(128, np.int32)
    launched[4096](cp.asarray(values), out)
    assert out.get().tolist() == values.reshape(128, 32).sum(axis=1).tolist()
    return out.get().tolist()


def sass(launched: Kernel) -> list[str]:
    """The SASS instructions of what `launched` compiled to, in order."""
    text = launched.dispatcher.inspect_sass(next(iter(launched.dispatcher.overloads)))
    found = re.finditer(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][\w.]*)", text)
    return [match[1] for match in found if match[1] != "NOP"]


def compiles(caplog: pytest.LogCaptureFixture) -> int:
    return sum("compiled cub" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("compiler", Compiler)
def test_cub_warp_sum_inlines_into_the_sass_of_patos_warp_sum(
    cache: Path, compiler: Compiler
) -> None:
    """A C++ body linked as LTO IR is no call: CUB's warp sum is the `redux.sync` patos writes.

    NVRTC compiles the unit in the process, and nvcc as it would a library NVRTC cannot parse.
    """
    _, sums = unit(compiler)

    assert summed(sums) == summed(patos_sums)
    assert sass(sums) == sass(patos_sums)
    assert not any(instruction.startswith("CALL") for instruction in sass(sums))


def test_a_unit_compiles_once_a_host_and_is_then_linked_from_the_cache(
    cache: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A second unit of the same source, as a later process makes, compiles nothing."""
    caplog.set_level(logging.INFO, logger="patos.cuda.typed.cxx")
    for _ in range(2):
        summed(unit()[1])

    assert compiles(caplog) == 1
    assert [entry.suffix for entry in sorted(cache.iterdir())] == [".ltoir", ".sha256"]


def test_a_unit_compiled_with_another_macro_is_another_entry(
    cache: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every flag helps decide a unit's bytes, so a macro makes a cache entry of its own."""
    caplog.set_level(logging.INFO, logger="patos.cuda.typed.cxx")
    for defines in (["NDEBUG"], ["NDEBUG", "PATOS_FLAGS=1"]):
        summed(unit(defines=defines)[1])

    assert compiles(caplog) == 2
    assert len(list(cache.glob("cub-sm*.ltoir"))) == 2


def test_a_damaged_entry_compiles_again_and_a_late_body_is_refused(
    cache: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Damaged bytes are compiled again, and a body declared after its unit compiled refused.

    A body declared late would link nowhere, since the unit's kernels hold the earlier object.
    """
    caplog.set_level(logging.INFO, logger="patos.cuda.typed.cxx")
    cub, sums = unit()
    summed(sums)
    (entry,) = cache.glob("cub-sm*.ltoir")
    entry.write_bytes(b"damaged")
    summed(unit()[1])

    assert compiles(caplog) == 2
    assert entry.read_bytes() != b"damaged"
    with pytest.raises(RuntimeError, match="late: declared after cub compiled"):
        cub(_REDUCE)(late)


def late(value: i32) -> i32:
    """A stub its unit receives after compiling."""
    raise NotImplementedError
