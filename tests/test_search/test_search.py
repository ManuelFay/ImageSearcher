import unittest
import os
import logging

from image_searcher.search.search import Search


class TestSearch(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(filename=None, level=logging.INFO)
        self.image_dir_path = "./tests/test_data"
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

    def test_searcher_tags(self):
        self.searcher = Search(image_dir_path=self.image_dir_path, include_faces=True)
        ranked_images = self.searcher.rank_images("A photo of a fast vehicle. #photo")
        self.assertIsInstance(ranked_images, list)

    def test_searcher_tags_group(self):
        self.searcher = Search(image_dir_path=self.image_dir_path, include_faces=True)
        ranked_images = self.searcher.rank_images("A photo of a fast vehicle. #group")
        self.assertIsInstance(ranked_images, list)

    def test_query_parser(self):
        self.searcher = Search(image_dir_path=self.image_dir_path)
        query, tags = self.searcher.parse_query("A photo of a fast vehicle.")
        self.assertIsInstance(query, str)
        self.assertIsInstance(tags, list)
        self.assertTrue(len(tags) == 0)

        query, tags = self.searcher.parse_query("A photo of a fast vehicle. #car #ootd #speed")
        self.assertIsInstance(query, str)
        self.assertIsInstance(tags, list)
        self.assertTrue(len(tags) == 3)

        query, tags = self.searcher.parse_query("#car #ootd #speed")
        self.assertIsInstance(query, str)
        self.assertIsInstance(tags, list)
        self.assertTrue(len(tags) == 3)

        query, tags = self.searcher.parse_query("")
        self.assertIsInstance(query, str)
        self.assertIsInstance(tags, list)
        self.assertTrue(len(tags) == 0)
