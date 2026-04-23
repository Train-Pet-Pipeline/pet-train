"""Tests for dual peer-dep guards in register_all() (F2, DEVELOPMENT_GUIDE §11.3 Mode B)."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_register_all_raises_when_pet_schema_missing(monkeypatch) -> None:
    """register_all() must raise RuntimeError with 'pet-schema' in message when absent."""
    monkeypatch.setitem(sys.modules, "pet_schema", None)

    import pet_train.plugins._register as register_mod

    importlib.reload(register_mod)

    with pytest.raises(RuntimeError, match="pet-schema"):
        register_mod.register_all()


def test_register_all_raises_when_pet_infra_missing(monkeypatch) -> None:
    """register_all() must raise RuntimeError with 'pet-infra' in message when absent."""
    monkeypatch.setitem(sys.modules, "pet_infra", None)

    import pet_train.plugins._register as register_mod

    importlib.reload(register_mod)

    with pytest.raises(RuntimeError, match="pet-infra"):
        register_mod.register_all()
