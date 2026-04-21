"""Entry-point target for pet-infra's plugin discovery.

pet-infra scans ``[project.entry-points."pet_infra.plugins"]`` and calls the
registered callable (named ``register_all``, matching pet-infra's own
convention — see ``pet_infra._register``) at CLI startup to trigger the
``@TRAINERS.register_module`` side-effects in plugin modules.
"""

from __future__ import annotations


def register_all() -> None:
    """Import pet-train plugin modules to trigger trainer registration side-effects."""
    try:
        import pet_infra  # noqa: F401  # peer-dep guard
    except ImportError as e:
        raise RuntimeError(
            "pet-train v2 requires pet-infra. Install via matrix row 2026.07."
        ) from e

    # Trainer plugin imports (P1-D/E/F fill these with actual @TRAINERS.register_module classes)
    from pet_train.plugins import llamafactory_dpo, llamafactory_sft, tiny_test  # noqa: F401
