import cv2
import pandas as pd

from config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS


class DatasetIndexer:
    """Scan image folders and create a dataframe with image paths and labels."""

    def __init__(self, data_dir=RAW_DATA_DIR) -> None:
        self.data_dir = data_dir

    def build_dataframe(self) -> pd.DataFrame:
        """Return image file paths, labels, width, height, and channels."""
        records = []

        for file_path in self.data_dir.rglob("*"):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            image = cv2.imread(str(file_path))

            if image is None:
                continue

            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            label = file_path.parent.name

            records.append(
                {
                    "file_path": str(file_path),
                    "label": label,
                    "width": width,
                    "height": height,
                    "channels": channels,
                }
            )

        return pd.DataFrame(records)