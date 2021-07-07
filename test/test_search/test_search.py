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
        self.searcher = Search(image_dir_path=self.image_dir_path)

    def tearDown(self) -> None:
        embed_path = os.path.join(self.image_dir_path, "stored_embeddings.pickle")
        if os.path.isfile(embed_path):
            logging.info(f"Removing {embed_path}")
            os.remove(embed_path)

    def test_searcher(self):
        ranked_images = self.searcher.rank_images("A photo of a fast vehicle.")
        print(ranked_images)

        ranked_images = self.searcher.rank_images("A photo of a bird.")
        print(ranked_images)
