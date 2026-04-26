"""Git-SHA lineage helpers for ModelCard provenance (F024 fix).

Replaces the legacy per-plugin ``_collect_git_shas`` static methods that
hardcoded ``{"pet_train": <CWD HEAD>}``. The hardcoded form had two structural
bugs that together broke ``pet run --replay`` drift detection:

  1. CWD's ``git rev-parse HEAD`` returns whichever repo the orchestrator
     happens to run in (typically pet-infra), but the key was always
     ``"pet_train"`` — wrong value under wrong key.
  2. ``pet_infra.replay.check_git_drift`` derives keys from sibling directory
     names (hyphenated, e.g. ``"pet-train"``), so the underscore key from
     the legacy plugin form never matched on lookup → drift never fired.

This module walks sibling repos under the monorepo root (same logic as
``pet_infra.replay._current_git_shas``) and returns ``{<dir-name>: <sha>}``
with hyphenated keys aligned with the replay-side scan.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def collect_git_shas() -> dict[str, str]:
    """Return ``{<repo-dir-name>: <HEAD sha>}`` for every sibling repo found.

    Walks up from this module file to find the monorepo root, then scans for
    sibling directories containing a ``.git`` entry. Each sibling's HEAD SHA is
    captured under its hyphenated directory name (matching the format used by
    :func:`pet_infra.replay._current_git_shas` so drift detection compares
    apples to apples).

    Returns:
        Mapping of repo directory name (hyphenated, e.g. ``"pet-train"``,
        ``"pet-schema"``) to its HEAD SHA. Empty dict when the repo root
        cannot be located (CI / installed-package contexts).
    """
    try:
        # Layout: <root>/pet-train/src/pet_train/lineage.py
        # parents indexing (Path.parents is 0-based on the IMMEDIATE parent dir):
        #   parents[0]=src/pet_train  parents[1]=src  parents[2]=pet-train  parents[3]=root
        # NB. pet_infra.replay.py used parents[4] (off-by-one — this is half of F024).
        module_file = Path(__file__).resolve()
        root = module_file.parents[3]
    except IndexError:
        log.debug("collect_git_shas: file not under expected monorepo layout")
        return {}

    siblings = [
        p for p in root.iterdir()
        if p.is_dir() and (p / ".git").exists()
    ]
    if not siblings:
        log.debug("collect_git_shas: no sibling repos at %s", root)
        return {}

    shas: dict[str, str] = {}
    for sibling in siblings:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(sibling),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            shas[sibling.name] = sha
        except (subprocess.CalledProcessError, OSError):
            # Non-fatal: skip repos where git fails (submodules, missing git, etc.)
            continue
    return shas
