from pathlib import Path

import cv2
import pandas as pd

from config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS
from models.records import ImageRecord


class DatasetIndexer:
    """Scan image folders and build a table of images and labels."""

    def __init__(self, data_dir: Path = RAW_DATA_DIR) -> None:
        # Folder that holds all class sub-folders (each sub-folder = one class).
        self.data_dir = data_dir

    def build_records(self) -> list[ImageRecord]:
        """Walk through the dataset folder and return one record per image."""
        records: list[ImageRecord] = []

        # Look at every file in every sub-folder.
        for file_path in self.data_dir.rglob("*"):
            # Skip anything that is not a supported image type.
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Try to read the image. If it fails, just skip the file.
            image = cv2.imread(str(file_path))
            if image is None:
                continue

            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1

            # The label is the name of the folder that holds the image.
            label = file_path.parent.name

            records.append(
                ImageRecord(
                    file_path=file_path,
                    label=label,
                    width=width,
                    height=height,
                    channels=channels,
                )
            )

        return records

    def build_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame view of the dataset records."""
        # Convert every ImageRecord to a dictionary, then to a DataFrame.
        records = self.build_records()
        return pd.DataFrame([record.to_dict() for record in records])
