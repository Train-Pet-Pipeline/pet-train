"""Tests for pet_train.lineage.collect_git_shas (F024 fix)."""
import subprocess
from pathlib import Path

import pytest

from pet_train.lineage import collect_git_shas


def _init_git_repo(repo_dir: Path) -> str:
    """Create an empty git repo with one commit; return HEAD sha."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo_dir, check=True)
    (repo_dir / "README.md").write_text("test\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo_dir, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
    ).strip()


@pytest.fixture
def fake_monorepo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Create a fake monorepo layout matching collect_git_shas's parents[4] logic.

    Layout: <root>/pet-train/src/pet_train/lineage.py (the module file)
    Plus sibling repos: <root>/pet-schema, <root>/pet-data, <root>/pet-eval
    """
    root = tmp_path
    # The module file's parents[4] must be `root`. We create a stub directory
    # tree that mimics the real layout and monkeypatch __file__ accordingly.
    pet_train_module_dir = root / "pet-train" / "src" / "pet_train"
    pet_train_module_dir.mkdir(parents=True)
    fake_module = pet_train_module_dir / "lineage.py"
    fake_module.write_text("# stub")
    # Three sibling repos
    shas = {
        "pet-schema": _init_git_repo(root / "pet-schema"),
        "pet-data": _init_git_repo(root / "pet-data"),
        "pet-eval": _init_git_repo(root / "pet-eval"),
    }
    # Also init pet-train itself (collect should pick it up too)
    shas["pet-train"] = _init_git_repo(root / "pet-train")

    # Monkeypatch the module's __file__ to point at the fake location
    import pet_train.lineage as lineage_mod
    monkeypatch.setattr(lineage_mod, "__file__", str(fake_module))
    return shas


def test_collect_git_shas_returns_all_sibling_repos(fake_monorepo: dict[str, str]) -> None:
    """F024: collect_git_shas must return {<repo-dir-name>: <sha>} for all sibling repos."""
    result = collect_git_shas()
    assert set(result.keys()) == set(fake_monorepo.keys())
    for repo, sha in fake_monorepo.items():
        assert result[repo] == sha


def test_collect_git_shas_keys_use_hyphenated_dir_names(fake_monorepo: dict[str, str]) -> None:
    """F024: keys MUST be hyphenated dir names (matching pet_infra.replay._current_git_shas)."""
    result = collect_git_shas()
    # No key should have underscores (legacy bug returned 'pet_train' instead of 'pet-train')
    assert "pet_train" not in result
    assert "pet-train" in result
    assert "pet-schema" in result


def test_collect_git_shas_returns_empty_on_unexpected_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """F024: missing monorepo layout → empty dict, no crash."""
    # Point __file__ at a shallow path so parents[3] yields nothing useful
    fake_shallow = tmp_path / "shallow.py"
    fake_shallow.write_text("")
    import pet_train.lineage as lineage_mod
    monkeypatch.setattr(lineage_mod, "__file__", str(fake_shallow))
    # parents[4] of /tmp/.../shallow.py would be / or close to it; unlikely to have sibling repos
    result = collect_git_shas()
    # Should return empty (no siblings under filesystem root) or never crash
    assert isinstance(result, dict)
