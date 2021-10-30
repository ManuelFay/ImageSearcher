import unittest

from image_searcher.interfaces.image_loader import ImageLoader


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.loader = ImageLoader(image_dir_path="./tests/test_data")

    def test_loader(self):
        images = self.loader.search_tree()
        self.assertTrue(len(images) > 5)
        self.assertIsInstance(images, list)
