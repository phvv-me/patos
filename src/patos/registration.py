"""Portable source registration, shared by independent research projects.

A registration binds one node to the exact set of sources that govern a run. The seal
covers the node's STATEMENT and the whole bytes of every other registered source, so a
node can settle, change its `status`, gain a date and grow a ledger or a log without
breaking the seal that pins the claim it was registered under.
"""

import hashlib
from collections.abc import Iterable
from pathlib import Path

from .bases import FrozenModel

_FENCE = b"---"
_HEADING = b"## "
_SETTLEMENT_HEADINGS = (b"## evidence", b"## ledger", b"## log")


def repository_path(repository: Path, path: Path) -> tuple[Path, str]:
    """Resolve one source and its repository-relative POSIX path.

    repository: the repository root every seal is taken under.
    path: an absolute path, or one relative to that root.
    """
    root = repository.resolve()
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        raise ValueError(f"{resolved} is outside repository {root}") from None
    return resolved, relative


def lf_bytes(path: Path) -> bytes:
    """Read bytes with Git's portable CRLF representation normalized to LF."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def statement_bytes(path: Path) -> bytes:
    """The sealed statement of a node file, LF-normalized.

    The statement runs from the first `## ` heading after the front matter up to, but
    excluding, the first `## Evidence`, `## Ledger` or `## Log` heading, matched without
    regard to case. Front matter and the settlement sections stay outside the seal, which
    is what lets a settled node rewrite `status` and append to its log. A file with no
    front matter, or with no `## ` heading at all, seals its whole body.

    path: the node file to read.
    """
    _, body = _front_matter_and_body(path)
    start = next((index for index, line in enumerate(body) if line.startswith(_HEADING)), 0)
    end = next(
        (index for index, line in enumerate(body[start:], start) if _settles(line)),
        len(body),
    )
    return b"".join(body[start:end])


def sha256_text_file(path: Path) -> str:
    """Return a checkout-independent digest of one registered UTF-8 text file.

    Git may materialize tracked text with CRLF on Windows and LF on POSIX. Sealing Git's
    canonical LF representation makes the identity survive either checkout.
    """
    return hashlib.sha256(lf_bytes(path)).hexdigest()


def sha256_text_sources(paths: Iterable[Path], *, repository: Path) -> str:
    """Hash one canonical source set by POSIX path and whole LF-normalized text.

    paths: the sources to seal, in any order.
    repository: the root every path is taken relative to.
    """
    return _seal((relative, lf_bytes(path)) for path, relative in _resolved(repository, paths))


def _settles(line: bytes) -> bool:
    """Whether this line is the heading that closes the statement."""
    return line.lower().startswith(_SETTLEMENT_HEADINGS)


def _front_matter_and_body(path: Path) -> tuple[tuple[str, ...], tuple[bytes, ...]]:
    """Split one node file into its front-matter fields and its body lines.

    A file whose first line is not a `---` fence has no front matter and is body only.
    Fields come back decoded and without their newline, the body keeps its raw LF bytes.
    """
    lines = lf_bytes(path).splitlines(keepends=True)
    if not lines or lines[0].rstrip(b"\n") != _FENCE:
        return (), tuple(lines)
    closing = next(
        (index for index in range(1, len(lines)) if lines[index].rstrip(b"\n") == _FENCE),
        None,
    )
    if closing is None:
        raise RuntimeError(f"the registered node has unclosed front matter: {path}")
    fields = tuple(line.rstrip(b"\n").decode("utf-8") for line in lines[1:closing])
    return fields, tuple(lines[closing + 1 :])


def _resolved(repository: Path, paths: Iterable[Path]) -> tuple[tuple[Path, str], ...]:
    """Every source resolved, checked for duplicates and existence, ordered by relative path."""
    sources = tuple(repository_path(repository, path) for path in paths)
    if len({path for path, _ in sources}) != len(sources):
        raise ValueError("registered sources contain duplicate paths")
    missing = [relative for path, relative in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"registered sources do not exist: {missing}")
    return tuple(sorted(sources, key=lambda item: item[1]))


def _seal(sources: Iterable[tuple[str, bytes]]) -> str:
    """Hash `relpath \\0 sealed bytes \\0` over an already ordered source set."""
    digest = hashlib.sha256()
    for relative, payload in sources:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


class SourceRegistration(FrozenModel):
    """The registered node and complete ordered set of sources governing a run."""

    repository: Path
    node: Path
    sources: tuple[Path, ...]
    seal_key: str = "registration_sha256"

    def digest(self) -> str:
        """Hash every registered source, the node contributing its statement alone."""
        node, _ = repository_path(self.repository, self.node)
        return _seal(
            (relative, statement_bytes(path) if path == node else lf_bytes(path))
            for path, relative in self._paths()
        )

    def verify(self) -> str:
        """Return the source seal after checking it against the sources on disk.

        The node's `status` is deliberately not read: the seal covers the statement, so a
        node that has settled still verifies against the claim it was registered under.
        """
        node, _ = repository_path(self.repository, self.node)
        prefix = f"{self.seal_key}: "
        fields, _ = _front_matter_and_body(node)
        seals = [field.removeprefix(prefix) for field in fields if field.startswith(prefix)]
        if len(seals) != 1:
            raise RuntimeError(f"the node must carry exactly one {self.seal_key} field")
        expected = self.digest()
        if seals[0] != expected:
            raise RuntimeError("the registered source seal drifted")
        return expected

    def _paths(self) -> tuple[tuple[Path, str], ...]:
        """The registered sources, ordered, with the node's membership enforced."""
        sources = _resolved(self.repository, self.sources)
        node, node_relative = repository_path(self.repository, self.node)
        if all(path != node for path, _ in sources):
            raise ValueError(f"registered sources do not include node {node_relative}")
        return sources
