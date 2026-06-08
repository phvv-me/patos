import importlib

import pytest

import patos
from patos._catalog import PATOS, Pato


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


@pytest.mark.parametrize("name", sorted(PATOS))
def test_catalog_matches_the_package_surface(name: str) -> None:
    """Every catalogued pato's module and exports resolve and are surfaced on the package."""
    pato = PATOS[name]
    assert isinstance(pato, Pato)
    module = importlib.import_module(f"patos.{pato.module.removesuffix('.py')}")
    for export in pato.exports:
        assert hasattr(module, export)
        assert export in patos.__all__
