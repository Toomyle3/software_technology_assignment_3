import cv2
import pandas as pd
from pathlib import Path
from records import ImageRecord


class DatasetIndexer:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)

    def build_dataframe(self):
        records = []

        for class_folder in self.dataset_dir.iterdir():
            if class_folder.is_dir():
                label = class_folder.name

                for image_path in class_folder.iterdir():
                    if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        image = cv2.imread(str(image_path))

                        if image is not None:
                            height, width, channels = image.shape

                            record = ImageRecord(
                                file_path=str(image_path),
                                label=label,
                                width=width,
                                height=height,
                                channels=channels
                            )

                            records.append(record)

        return pd.DataFrame([record.__dict__ for record in records])
    