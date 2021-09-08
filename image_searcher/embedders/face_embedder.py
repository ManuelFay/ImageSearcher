import logging
from typing import List
from PIL import Image
import numpy as np

try:
    import face_recognition
    _FACE_RECOGNITION_LOADED = True
except ImportError:
    _FACE_RECOGNITION_LOADED = False


class FaceEmbedder:
    def __init__(self, model: str = "large", num_jitters: int = 5):
        logging.info("Loading FaceEmbedder")
        self.model = model
        self.num_jitters = num_jitters

    def embed_image(self, image: Image.Image) -> List:
        if _FACE_RECOGNITION_LOADED:
            encodings = face_recognition.face_encodings(np.array(image), model=self.model, num_jitters=self.num_jitters)
            return encodings
        return []
