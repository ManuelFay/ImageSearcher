from dataclasses import dataclass
from typing import Optional


@dataclass
class FlaskConfig:
    image_dir_path: str
    save_path: Optional[str] = None
    traverse: bool = True
    reindex: bool = True
    include_faces: bool = False
    n: int = 42

    port: Optional[int] = None
    host: Optional[str] = None
    debug: Optional[bool] = None
    threaded: Optional[bool] = True
