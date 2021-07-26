import unittest
import os
import logging

from image_searcher.search.search import Search


class TestSearch(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(filename=None, level=logging.INFO)
        self.image_dir_path = "/home/manu/perso/ImageSearcher/data/"
        # self.image_dir_path = "/home/manu/Downloads/"
        self.tearDown()

    def tearDown(self) -> None:
        embed_path = os.path.join(self.image_dir_path, "stored_embeddings.pickle")
        if os.path.isfile(embed_path):
            logging.info(f"Removing {embed_path}")
            os.remove(embed_path)

    def test_searcher(self):
        self.searcher = Search(image_dir_path=self.image_dir_path)
        ranked_images = self.searcher.rank_images("A photo of a fast vehicle.")
        self.assertIsInstance(ranked_images, list)

        self.searcher = Search(image_dir_path=None, save_path=self.image_dir_path)
        ranked_images = self.searcher.rank_images("A photo of a fast vehicle.")
        self.assertIsInstance(ranked_images, list)
        print(ranked_images)

