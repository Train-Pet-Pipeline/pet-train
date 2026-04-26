"""pet-train: SFT/DPO training and audio CNN for smart pet feeder pipeline."""

from importlib.metadata import PackageNotFoundError, version

__version__ = "2.2.5"

try:
    __version__ = version("pet-train")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "2.2.5"
