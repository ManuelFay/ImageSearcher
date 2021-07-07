from dataclasses import dataclass
from typing import Optional


@dataclass
class FlaskConfig:
    image_dir_path: str
    traverse: bool = True

    port: Optional[int] = None
    host: Optional[str] = None
    debug: Optional[bool] = None
    threaded: Optional[bool] = True
