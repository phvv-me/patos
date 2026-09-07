"""Portable source registration, shared by independent research projects."""

import hashlib
from pathlib import Path

from .bases import FrozenModel

_REGISTRATION_PLACEHOLDER = "0" * 64


def _repository_path(repository: Path, path: Path) -> tuple[Path, str]:
    root = repository.resolve()
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        raise ValueError(f"{resolved} is outside repository {root}") from None
    return resolved, relative


def _lf_bytes(path: Path) -> bytes:
    """Read bytes with Git's portable CRLF representation normalized to LF."""
    return path.read_bytes().replace(b"\r\n", b"\n")


class SourceRegistration(FrozenModel):
    """The registered node and complete ordered set of sources governing a run."""

    repository: Path
    node: Path
    sources: tuple[Path, ...]
    status: str = "registered"
    seal_key: str = "registration_sha256"

    def _paths(self) -> tuple[tuple[Path, str], ...]:
        paths = [_repository_path(self.repository, path) for path in self.sources]
        node, node_relative = _repository_path(self.repository, self.node)
        if all(path != node for path, _ in paths):
            raise ValueError(f"registered sources do not include node {node_relative}")
        if len({path for path, _ in paths}) != len(paths):
            raise ValueError("registered sources contain duplicate paths")
        missing = [relative for path, relative in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"registered sources do not exist: {missing}")
        return tuple(sorted(paths, key=lambda item: item[1]))

    def _registered_bytes(self, path: Path) -> bytes:
        payload = _lf_bytes(path)
        node, _ = _repository_path(self.repository, self.node)
        if path != node:
            return payload
        prefix = f"{self.seal_key}: ".encode()
        lines = payload.splitlines(keepends=True)
        matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(f"the node must contain exactly one {self.seal_key} field")
        index = matches[0]
        ending = b"\n" if lines[index].endswith(b"\n") else b""
        lines[index] = prefix + _REGISTRATION_PLACEHOLDER.encode() + ending
        return b"".join(lines)

    def digest(self) -> str:
        """Hash relative POSIX paths and LF-normalized registered bytes."""
        digest = hashlib.sha256()
        for path, relative in self._paths():
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self._registered_bytes(path))
            digest.update(b"\0")
        return digest.hexdigest()

    def verify(self) -> str:
        """Return the source seal after checking node status and source identity."""
        node, _ = _repository_path(self.repository, self.node)
        lines = _lf_bytes(node).decode("utf-8").splitlines()
        if not lines or lines[0] != "---":
            raise RuntimeError("the registered node has no front matter")
        try:
            end = lines.index("---", 1)
        except ValueError:
            raise RuntimeError("the registered node has unclosed front matter") from None
        header = lines[1:end]
        if header.count(f"status: {self.status}") != 1:
            raise RuntimeError(f"the experiment node does not have status {self.status!r}")
        seals = [
            line.removeprefix(f"{self.seal_key}: ")
            for line in header
            if line.startswith(f"{self.seal_key}: ")
        ]
        expected = self.digest()
        if seals != [expected]:
            raise RuntimeError("the registered source seal drifted")
        return expected
