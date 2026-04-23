"""Tests for pet_train.__version__ parity with installed package metadata (F10)."""

from __future__ import annotations

import importlib.metadata

import pet_train


def test_version_attribute_exists() -> None:
    """pet_train.__version__ must be defined."""
    assert hasattr(pet_train, "__version__")
    assert isinstance(pet_train.__version__, str)
    assert pet_train.__version__


def test_version_matches_installed_metadata() -> None:
    """pet_train.__version__ must match importlib.metadata.version('pet-train')."""
    installed = importlib.metadata.version("pet-train")
    assert pet_train.__version__ == installed, (
        f"__version__ {pet_train.__version__!r} != installed metadata {installed!r}"
    )
