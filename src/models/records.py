"""Data classes used across the project. ImageRecord is the simple
value object that represents one indexed image."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageRecord:
    """Hold the basic info for one macroinvertebrate image."""

    # Where the image file is on disk.
    file_path: Path
    # The class name (we use the parent folder name as the label).
    label: str
    # Image width in pixels.
    width: int
    # Image height in pixels.
    height: int
    # 1 for grayscale, 3 for colour images.
    channels: int

    def to_dict(self) -> dict:
        """Turn this record into a plain dictionary (used by pandas)."""
        return {
            "file_path": str(self.file_path),
            "label": self.label,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
        }
