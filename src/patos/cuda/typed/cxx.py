"""Device functions whose body is C++ from a header library, declared by an annotated stub: `ptx`'s
sibling for what one PTX block cannot hold, such as CUB's collectives.

One `Cxx` is one translation unit of one library, its prelude and every body declared into it. The
first kernel calling one of them compiles the unit to LTO IR for the GPU at hand and links it, so
the call inlines as a device function's does. The compiled unit is cached in the user's cache
directory under the digest of what decides its bytes (the source, the headers, the compiler and
every flag it runs with, which holds the architecture), so a host compiles each unit once, and an
entry whose bytes no longer match the digest recorded beside them is compiled again rather than
linked.

    cub = Cxx(cccl, "#include <cub/warp/warp_reduce.cuh>", name="cub")

    @cub('''
        using Reduce = cub::WarpReduce<cuda::std::uint32_t>;
        __shared__ Reduce::TempStorage storage[32];
        return Reduce(storage[threadIdx.x / 32]).Sum(value);
    ''')
    def warp_total(value: u32) -> u32:
        '''The sum of `value` across the warp, in lane 0.'''
        raise NotImplementedError

A stub's parameters are scalars or lanes, its return one of these or None. NVRTC compiles a unit
in the process; a library whose headers reach host-only code declares `Compiler.NVCC`, and the
environment's nvcc compiles it instead.

CCCL comes from its wheel. A library with no package, such as cuCollections (`cuco`), is pinned
to one commit in `libraries.toml` and fetched into the user's cache on its first compile.
"""

import hashlib
import importlib.metadata
import inspect
import io
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from enum import StrEnum, auto
from functools import cache, partial
from pathlib import Path
from types import FunctionType

from cuda.bindings import nvrtc
from cuda.cccl import get_include_paths
from cuda.core import Program, ProgramOptions
from llvmlite import ir
from numba import cuda, types
from numba.core.typing import templates
from numba.cuda import cgutils

from ...bases import FrozenModel
from .intrinsics import as_intrinsic, read_stub
from .lanes import LaneType

logger = logging.getLogger(__name__)

# The C++ spelling of each type a stub may declare.
_SPELLED = {
    types.int8: "cuda::std::int8_t", types.uint8: "cuda::std::uint8_t",
    types.int16: "cuda::std::int16_t", types.uint16: "cuda::std::uint16_t",
    types.int32: "cuda::std::int32_t", types.uint32: "cuda::std::uint32_t",
    types.int64: "cuda::std::int64_t", types.uint64: "cuda::std::uint64_t",
    types.float32: "float", types.float64: "double", types.boolean: "bool", types.none: "void",
}  # fmt: skip


class Compiler(StrEnum):
    """What turns a unit into LTO IR: NVRTC in the process, or nvcc for host-reaching headers."""

    NVRTC = auto()
    NVCC = auto()

    @property
    def suffix(self) -> str:
        """What a compiled unit's file ends in: NVRTC writes bare LTO IR, nvcc a fatbin of it."""
        return "ltoir" if self is Compiler.NVRTC else "fatbin"

    def compiled(self, name: str, headers: Headers, source: str, arch: int) -> bytes:
        """The LTO IR of the unit `name` holding `source`, for `sm_<arch>`."""
        match self:
            case Compiler.NVRTC:
                program = Program(source, "c++", options=self._program(headers, arch, name))
                return bytes(program.compile("ltoir").code)
            case Compiler.NVCC:
                return self._nvcc(name, self.flags(headers, arch), source)

    def flags(self, headers: Headers, arch: int) -> list[str]:
        """Every flag a unit for `sm_<arch>` compiles with, which the cache key holds."""
        match self:
            case Compiler.NVRTC:
                return [flag.decode() for flag in self._program(headers, arch).as_bytes("nvrtc")]
            case Compiler.NVCC:
                return [
                    "-std=c++17", "-rdc=true", "-O3", "-fatbin", "--extended-lambda",
                    f"-gencode=arch=compute_{arch},code=lto_{arch}",
                    *(f"-D{define}" for define in headers.defines),
                    *(f"-I{path}" for path in headers.include),
                ]  # fmt: skip

    def identity(self) -> str:
        """The compiler's version, or nvcc's binary by path, size and time.

        Either moves with any toolkit change, and neither runs a subprocess on a cached path.
        """
        match self:
            case Compiler.NVRTC:
                _, major, minor = nvrtc.nvrtcVersion()
                return f"nvrtc {major}.{minor}"
            case Compiler.NVCC:
                found = _NVCC.resolve()
                return f"nvcc {found} {found.stat().st_size} {found.stat().st_mtime_ns}"

    @staticmethod
    def _nvcc(name: str, flags: list[str], source: str) -> bytes:
        """The fatbin of LTO IR the environment's nvcc compiles `source` to.

        The environment's `bin` leads the search path, so its own `ptxas`, `nvlink` and host
        compiler serve nvcc before any the system has.
        """
        search = os.pathsep.join([str(_NVCC.parent), os.environ.get("PATH", "")])
        with tempfile.TemporaryDirectory() as folder:
            written, output = Path(folder, f"{name}.cu"), Path(folder, f"{name}.fatbin")
            written.write_text(source)
            command = [str(_NVCC), *flags, str(written), "-o", str(output)]
            built = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=os.environ | {"PATH": search},
            )
            if built.returncode:
                raise RuntimeError(f"nvcc could not compile {name}:\n{built.stderr}")
            return output.read_bytes()

    @staticmethod
    def _program(headers: Headers, arch: int, name: str = "") -> ProgramOptions:
        return ProgramOptions(
            name=name, arch=f"sm_{arch}", std="c++17", relocatable_device_code=True,
            link_time_optimization=True, define_macro=headers.defines,
            include_path=[str(path) for path in headers.include],
        )  # fmt: skip


class Headers(FrozenModel):
    """A header library: its include directories, a digest of its headers' bytes, the macros it
    needs and the compiler that parses it."""

    name: str
    include: list[Path]
    digest: str
    compiler: Compiler = Compiler.NVRTC
    defines: list[str] = ["NDEBUG"]


def cccl() -> Headers:
    """CCCL as the `cuda-cccl` wheel carries it: libcu++, CUB and Thrust.

    The wheel's CUDA toolkit headers stay out, so each compiler reads its own; the digest is of
    the wheel's record, which lists every header's.
    """
    record = importlib.metadata.distribution("cuda-cccl").read_text("RECORD")
    if record is None:
        raise FileNotFoundError("cuda-cccl has no RECORD, so its headers have no digest to key on")
    return Headers(
        name="cccl",
        include=[Path(get_include_paths().libcudacxx)],
        digest=hashlib.sha256(record.encode()).hexdigest(),
    )


@cache
def cuco() -> Headers:
    """cuCollections at the commit `libraries.toml` pins, over CCCL from its wheel.

    nvcc compiles it, since its refs reach host-only headers NVRTC refuses (cuCollections#695).
    """
    pinned, base = Pinned.registered("cuco"), cccl()
    return Headers(
        name="cuco",
        include=[pinned.include(), *base.include],
        digest=hashlib.sha256(f"{pinned.headers_sha256}\0{base.digest}".encode()).hexdigest(),
        compiler=Compiler.NVCC,
    )


class Pinned(FrozenModel):
    """A header library pinned to one commit of its repository, as `libraries.toml` registers it.

    The archive is fetched into the user's cache on first use and refused unless it hashes as
    registered; the headers are hashed again in each process that compiles against them, so an
    edited or truncated copy is refused rather than compiled.
    """

    name: str
    repository: str
    commit: str
    archive_sha256: str
    headers_sha256: str

    @classmethod
    def registered(cls, name: str) -> Pinned:
        return cls(name=name, **tomllib.loads(_REGISTRY.read_text())[name])

    def include(self) -> Path:
        """The library's include directory in the user's cache, fetched when missing."""
        path = _cache() / "headers" / f"{self.name}-{self.commit[:12]}"
        if not path.is_dir():
            self._fetch(path)
        _checked(tree_digest(path), registered=self.headers_sha256, source=path)
        return path

    def _fetch(self, path: Path) -> None:
        """Unpack the archive's `include` into `path` whole.

        A copy another process placed first stays, and the digest check after any other failed
        rename refuses the missing folder.
        """
        url = f"{self.repository}/archive/{self.commit}.tar.gz"
        logger.info("fetching %s", url)
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = response.read()
        _checked(hashlib.sha256(archive).hexdigest(), registered=self.archive_sha256, source=url)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as folder:
            with tarfile.open(fileobj=io.BytesIO(archive)) as unpacked:
                unpacked.extractall(folder, filter="data")
            (top,) = Path(folder).iterdir()
            with suppress(OSError):
                (top / "include").rename(path)


def tree_digest(root: Path) -> str:
    """The SHA-256 of every file under `root`: its relative path, its size and its bytes."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            digest.update(f"{path.relative_to(root).as_posix()}\0{len(data)}\0".encode() + data)
    return digest.hexdigest()


def _checked(found: str, *, registered: str, source: Path | str) -> None:
    if found != registered:
        raise RuntimeError(
            f"{source} hashes {found}, and {_REGISTRY.name} registers {registered}; delete a "
            "cached copy to fetch it again"
        )


class Cxx:
    """One translation unit of a header library: its prelude and the bodies declared into it.

    headers: finds the library, called at the unit's first compile, so importing the bindings
        reads nothing.
    """

    def __init__(self, headers: Callable[[], Headers], prelude: str = "", *, name: str) -> None:
        self._headers = headers
        self.name = name
        self._parts = ["#include <cuda/std/cstdint>", prelude]
        self._ltoirs: dict[int, Path] = {}

    def __call__[F: FunctionType](self, body: str) -> Callable[[F], F]:
        """Make an annotated stub a device function running the C++ `body`."""

        def defined(stub: F) -> F:
            if self._ltoirs:
                raise RuntimeError(f"{stub.__qualname__}: declared after {self.name} compiled")
            signature, typed = read_stub(stub)
            symbol = f"patos_{self.name}_{stub.__name__}"
            names = inspect.signature(stub).parameters
            declared = ", ".join(
                f"{_spelled(kind)} {name}" for kind, name in zip(typed.args, names, strict=True)
            )
            self._parts.append(
                f'extern "C" __device__ {_spelled(typed.return_type)} {symbol}({declared})\n'
                f"{{\n{body}\n}}"
            )
            return as_intrinsic(stub, signature, typed, partial(self._called, symbol))

        return defined

    def ltoir(self) -> Path:
        """The unit's compiled LTO IR for the current GPU, from the cache when it holds it."""
        major, minor = cuda.get_current_device().compute_capability
        arch = 10 * major + minor
        if arch not in self._ltoirs:
            self._ltoirs[arch] = self._cached(arch)
        return self._ltoirs[arch]

    @staticmethod
    def _is_intact(path: Path) -> bool:
        """Whether the entry at `path` holds the bytes whose digest was recorded beside it."""
        recorded = path.with_suffix(".sha256")
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest() == recorded.read_text().strip()
        except FileNotFoundError:
            return False

    def _cached(self, arch: int) -> Path:
        """The cache entry of the unit for `sm_<arch>`, compiled when missing or damaged."""
        headers, source = self._headers(), "\n".join(self._parts)
        path = self._entry(headers, source, arch)
        if not self._is_intact(path):
            data = headers.compiler.compiled(self.name, headers, source, arch)
            logger.info("compiled %s for sm_%d with %s", self.name, arch, headers.compiler)
            _written(path, data)
            _written(path.with_suffix(".sha256"), hashlib.sha256(data).hexdigest().encode())
        return path

    def _called(
        self,
        symbol: str,
        context,
        builder: ir.IRBuilder,
        call: templates.Signature,
        arguments: list,
    ) -> ir.Value:
        """Call `symbol` in the unit, linked into the kernel being compiled."""
        context.active_code_library.add_linking_file(str(self.ltoir()))
        void = call.return_type == types.none
        returns = ir.VoidType() if void else context.get_value_type(call.return_type)
        kinds = [context.get_value_type(kind) for kind in call.args]
        callee = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(returns, kinds), symbol
        )
        value = builder.call(callee, arguments)
        return context.get_dummy_value() if void else value

    def _entry(self, headers: Headers, source: str, arch: int) -> Path:
        """Where the unit lives in the user's cache, named by all that decides its bytes."""
        compiler = headers.compiler
        decided = [source, headers.name, headers.digest, compiler.identity()]
        decided += compiler.flags(headers, arch)
        key = hashlib.sha256("\0".join(decided).encode()).hexdigest()[:32]
        return _cache() / "cxx" / f"{self.name}-sm{arch}-{key}.{compiler.suffix}"


# The environment's own nvcc, which the `cuda-nvcc` package puts beside its Python.
_NVCC = Path(sys.prefix, "bin", "nvcc")
_REGISTRY = Path(__file__).with_name("libraries.toml")


def _cache() -> Path:
    """patos's folder in the user's cache directory."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache", "patos")


def _spelled(kind: types.Type) -> str:
    return "cuda::std::uint32_t" if isinstance(kind, LaneType) else _SPELLED[kind]


def _written(path: Path, data: bytes) -> None:
    """Write `data` to `path` whole: another process reads either nothing or every byte."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as partial:
        partial.write(data)
    Path(partial.name).replace(path)
