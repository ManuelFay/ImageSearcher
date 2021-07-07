from dataclasses import dataclass


@dataclass
class RankedImage:
    image_path: str
    score: float
