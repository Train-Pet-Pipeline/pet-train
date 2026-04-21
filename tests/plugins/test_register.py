def test_register_all_succeeds_with_peer_dep() -> None:
    from pet_train.plugins._register import register_all

    register_all()  # should not raise — pet-infra is installed in the test env


def test_register_all_fails_without_pet_infra(monkeypatch) -> None:
    import builtins

    import pytest

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pet_infra":
            raise ImportError("simulated missing pet-infra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Must re-import the module so the guard re-runs under the monkeypatched import
    import importlib

    import pet_train.plugins._register as register_mod

    importlib.reload(register_mod)

    with pytest.raises(RuntimeError, match="pet-infra"):
        register_mod.register_all()
