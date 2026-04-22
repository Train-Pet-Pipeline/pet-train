# Changelog

All notable changes to pet-train are documented here.
Format follows Keep a Changelog; versions follow SemVer.

## [2.0.1] - 2026-04-22

Phase 4 P5-A-5 final cut. Folds in two carry-over fixes from prior phases.

### Fixed
- `pyproject.toml` version bumped from drift value `0.1.0` → `2.0.1`. The
  prior v2.0.0 / v2.0.0-rc1 tags shipped with `pyproject.version = "0.1.0"`
  due to a missed bump during the Phase 3A release; this re-aligns the
  declared package version with the matrix-published tag (P5-A-5).

### Changed
- CI peer-dep pin bumped from `pet-infra @ git+...@v2.3.0-rc1` → `@v2.5.0`
  (matrix 2026.09). Updated in both `ci.yml` and `peer-dep-smoke.yml` (P5-A-5).

### Removed
- W&B residue (P2-C-1, shipped to dev 2026-04-22 ahead of this release):
  - `wandb` block deleted from `params.yaml`.
  - `wandb/` entry removed from `.gitignore`.

  ClearML (orchestrator P0-B/C) is the sole experiment tracker.

## [2.0.0] - 2026-04-21

Phase 3A — initial plugin port (3 trainers + audio namespace + 4-step
peer-dep CI). See git history for details. Note: this tag shipped with a
declared `pyproject.version = "0.1.0"`, corrected in 2.0.1.
