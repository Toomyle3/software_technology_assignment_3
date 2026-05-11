from dataclasses import dataclass


@dataclass
class ImageRecord:
    file_path: str
    label: str
    width: int
    height: int
    channels: int
    