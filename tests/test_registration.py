from hashlib import sha256
from pathlib import Path

import pytest

from patos import SourceRegistration, sha256_text_file, sha256_text_sources, statement_bytes

_SEAL = "registration_sha256"

_NODE = """---
status: registered
date: 2026-09-10
registration_sha256: {seal}
---

# Title

## Statement

The registered claim.

## Log

- [who 2026-09-10] settled
"""


def _write(path: Path, text: str, newline: str = "\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return path


def _registration(repository: Path, text: str = _NODE.format(seal="0" * 64)) -> SourceRegistration:
    _write(repository / "run.py", "VALUE = 1\n")
    node = _write(repository / "experiment" / "node.md", text)
    return SourceRegistration(
        repository=repository,
        node=node,
        sources=(node, repository / "run.py"),
    )


def test_statement_stops_at_the_first_settlement_heading(tmp_path: Path) -> None:
    """The statement runs from the first `## ` heading to the first settlement heading."""
    node = _write(tmp_path / "node.md", _NODE.format(seal="0" * 64))

    assert statement_bytes(node) == b"## Statement\n\nThe registered claim.\n\n"


@pytest.mark.parametrize("heading", ["## Evidence", "## LEDGER", "## log entries", "## Log"])
def test_every_settlement_heading_closes_the_statement_in_any_case(
    tmp_path: Path,
    heading: str,
) -> None:
    """`Evidence`, `Ledger` and `Log` all close the statement, matched without regard to case."""
    node = _write(
        tmp_path / "node.md",
        f"---\nstatus: registered\n---\n\n## Claim\n\nOne.\n\n{heading}\n\nlater\n",
    )

    assert statement_bytes(node) == b"## Claim\n\nOne.\n\n"


def test_a_node_without_front_matter_or_headings_seals_its_whole_body(tmp_path: Path) -> None:
    """No front matter makes the whole file body, and no `## ` heading seals that whole body."""
    plain = _write(tmp_path / "plain.md", "just prose\nover two lines\n")
    headless = _write(tmp_path / "headless.md", "---\nstatus: settled\n---\n\nprose only\n")

    assert statement_bytes(plain) == b"just prose\nover two lines\n"
    assert statement_bytes(headless) == b"\nprose only\n"


def test_unclosed_front_matter_is_refused(tmp_path: Path) -> None:
    """A node whose front matter never closes has no statement to seal."""
    node = _write(tmp_path / "node.md", "---\nstatus: registered\n\n## Claim\n")

    with pytest.raises(RuntimeError, match="unclosed front matter"):
        statement_bytes(node)


def test_the_seal_survives_settlement_of_the_node(tmp_path: Path) -> None:
    """Rewriting `status`, adding a field and appending to the log leave the seal intact."""
    registration = _registration(tmp_path)
    seal = registration.digest()
    _write(tmp_path / "experiment" / "node.md", _NODE.format(seal=seal))
    assert registration.verify() == seal

    settled = (
        _NODE.format(seal=seal)
        .replace("status: registered", "status: settled")
        .replace("date: 2026-09-10", "date: 2026-09-10\naliases: [claim]")
        + "- [who 2026-09-11] evidence pinned\n"
    )
    _write(tmp_path / "experiment" / "node.md", settled)

    assert registration.digest() == seal
    assert registration.verify() == seal


def test_editing_the_statement_breaks_the_seal(tmp_path: Path) -> None:
    """The claim itself stays sealed, so any edit inside the statement is refused."""
    registration = _registration(tmp_path)
    seal = registration.digest()
    _write(
        tmp_path / "experiment" / "node.md",
        _NODE.format(seal=seal).replace("The registered claim.", "A different claim."),
    )

    with pytest.raises(RuntimeError, match="seal drifted"):
        registration.verify()


def test_verify_needs_exactly_one_seal_field(tmp_path: Path) -> None:
    """A node with no seal field, or with two, cannot be verified against one digest."""
    registration = _registration(tmp_path)
    node = tmp_path / "experiment" / "node.md"
    _write(node, _NODE.format(seal="0" * 64).replace(f"{_SEAL}: {'0' * 64}\n", ""))
    with pytest.raises(RuntimeError, match="exactly one"):
        registration.verify()

    _write(node, _NODE.format(seal="0" * 64).replace(f"{_SEAL}: ", f"{_SEAL}: 0\n{_SEAL}: "))
    with pytest.raises(RuntimeError, match="exactly one"):
        registration.verify()


def test_the_seal_is_identical_across_lf_and_crlf_checkouts(tmp_path: Path) -> None:
    """Git's Windows checkout newline must not move the seal."""
    registration = _registration(tmp_path)
    lf = registration.digest()
    for path in registration.sources:
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    assert registration.digest() == lf
    assert sha256_text_sources(registration.sources, repository=tmp_path)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_single_file_digest_uses_canonical_lf_bytes(tmp_path: Path, newline: str) -> None:
    """A file digest names the same source in either checkout representation."""
    path = _write(tmp_path / "run.py", "VALUE = 1\n", newline=newline)

    assert sha256_text_file(path) == sha256(b"VALUE = 1\n").hexdigest()


def test_registered_sources_must_be_whole_and_include_the_node(tmp_path: Path) -> None:
    """Duplicates, missing files and a source set without its own node are all refused."""
    registration = _registration(tmp_path)
    node = registration.node

    with pytest.raises(ValueError, match="duplicate"):
        SourceRegistration(repository=tmp_path, node=node, sources=(node, node)).digest()
    with pytest.raises(FileNotFoundError, match="do not exist"):
        SourceRegistration(
            repository=tmp_path, node=node, sources=(node, tmp_path / "absent.py")
        ).digest()
    with pytest.raises(ValueError, match="do not include node"):
        SourceRegistration(repository=tmp_path, node=node, sources=(tmp_path / "run.py",)).digest()
    with pytest.raises(ValueError, match="outside repository"):
        SourceRegistration(
            repository=tmp_path / "experiment", node=node, sources=(node, tmp_path / "run.py")
        ).digest()
