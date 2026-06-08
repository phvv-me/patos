import importlib

import pytest

import patos


@pytest.mark.parametrize("export", patos.__all__)
def test_every_export_is_importable(export: str) -> None:
    """Each name in __all__ is a real attribute on the package."""
    assert getattr(patos, export) is not None


def test_version_falls_back_when_metadata_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.metadata import PackageNotFoundError

    def absent(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("importlib.metadata.version", absent)
    reloaded = importlib.reload(patos)
    try:
        assert reloaded.__version__ == "0.0.0"
    finally:
        monkeypatch.undo()
        importlib.reload(patos)
