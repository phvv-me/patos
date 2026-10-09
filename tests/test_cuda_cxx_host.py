"""What a C++ unit depends on the host for: the architecture it targets, the compiler it runs and
the cache directory it is kept in."""

import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import pytest
from test_cuda_cxx import compiled_by, compiles, summed, unit

from patos.cuda.typed import Compiler
from patos.cuda.typed import cxx as cxx_module

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="launches need a GPU")


@pytest.mark.parametrize(
    ("compiler", "capability", "arch"),
    [
        (Compiler.NVRTC, (8, 6), "86"),
        (Compiler.NVRTC, (9, 0), "90a"),
        (Compiler.NVRTC, (12, 1), "121a"),
        (Compiler.NVCC, (9, 0), "90a"),
    ],
)
def test_a_unit_is_compiled_for_the_architecture_numba_cuda_compiles_the_kernels_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiler: Compiler,
    capability: tuple[int, int],
    arch: str,
) -> None:
    """Hopper and later get `sm_90a`, `sm_121a`, the target of numba-cuda's code; older GPUs none.

    The flags, which the cache key holds, name it for both compilers, and so does the entry.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    device = SimpleNamespace(compute_capability=capability)
    monkeypatch.setattr(cxx_module, "cuda", SimpleNamespace(get_current_device=lambda: device))
    cub, _ = unit(compiler)
    entry = cub.ltoir()

    flags = " ".join(compiler.flags(compiled_by(compiler, ["NDEBUG"]), int(arch.rstrip("a"))))
    wanted = f"sm_{arch}" if compiler is Compiler.NVRTC else f"compute_{arch},code=lto_{arch}"
    assert entry.name.startswith(f"cub-sm{arch}-")
    assert wanted in flags


def loading(library: Path) -> Callable[[str], SimpleNamespace]:
    """A stand-in for the loader of NVRTC that finds `library`."""
    return lambda _: SimpleNamespace(abs_path=str(library))


def test_the_nvrtc_identity_moves_with_a_patch_release_of_its_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The version NVRTC reports is its major and minor, which a patch release keeps."""
    identities = []
    for size in (10, 11):
        library = tmp_path / f"libnvrtc-{size}.so"
        library.write_bytes(b"x" * size)
        monkeypatch.setattr(cxx_module, "load_nvidia_dynamic_lib", loading(library))
        identities.append(Compiler.NVRTC.identity())

    assert identities[0] != identities[1]
    assert {identity.split()[1] for identity in identities} == {identities[0].split()[1]}


def test_a_cache_that_cannot_be_written_compiles_without_caching_and_says_so_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A unit compiles into a temporary folder, with one warning however many it compiles."""
    blocked = tmp_path / "blocked"
    blocked.touch()
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocked / "below"))
    caplog.set_level(logging.INFO, logger="patos.cuda.typed.cxx")

    with pytest.warns(RuntimeWarning) as said:
        for _ in range(2):
            summed(unit()[1])

    unwritable = [each for each in said if "compiling without caching" in str(each.message)]
    assert len(unwritable) == 1
    assert compiles(caplog) == 1
