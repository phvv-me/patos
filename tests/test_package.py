import importlib
import sys

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


def test_lazy_model_names_match_the_models_package() -> None:
    """`patos.<name>` resolves exactly the names `patos.models` exports, and nothing else."""
    models = importlib.import_module("patos.models")
    assert frozenset(models.__all__) == patos._MODELS
    assert all(getattr(patos, name) is getattr(models, name) for name in models.__all__)
    with pytest.raises(AttributeError):
        patos.NoSuchName  # noqa: B018


def test_core_imports_without_pydantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core patterns import and work with pydantic absent; only a model name needs it."""
    monkeypatch.setitem(sys.modules, "pydantic", None)
    for name in [name for name in sys.modules if name == "patos" or name.startswith("patos.")]:
        monkeypatch.delitem(sys.modules, name)
    core = importlib.import_module("patos")
    assert core.value_dispatch is not None
    with pytest.raises(ImportError):
        core.FrozenModel  # noqa: B018
    with pytest.raises(AttributeError):
        core.NoSuchName  # noqa: B018
