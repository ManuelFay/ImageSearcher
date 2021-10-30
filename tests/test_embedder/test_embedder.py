import unittest
from PIL import Image

from image_searcher.interfaces.image_loader import ImageLoader
from image_searcher.embedders.clip_embedder import ClipEmbedder


class TestEmbedder(unittest.TestCase):
    def setUp(self):
        self.loader = ImageLoader(image_dir_path="./tests/test_data", traverse=True)
        self.embedder = ClipEmbedder()

    def test_image_embedder(self):
        image_paths = self.loader.search_tree()[:5]
        inputs = self.embedder.embed_images([Image.open(file).convert("RGB") for file in image_paths])
        self.assertTrue(inputs.shape[1] == 512)

    def test_text_embedder(self):
        inputs = self.embedder.embed_text("A photo of a cat")
        self.assertTrue(inputs.shape[1] == 512)
