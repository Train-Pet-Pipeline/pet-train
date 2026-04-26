"""pet-train: SFT/DPO training and audio CNN for smart pet feeder pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pet-train")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
