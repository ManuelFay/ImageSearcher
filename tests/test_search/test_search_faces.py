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
        self.searcher = Search(image_dir_path=self.image_dir_path, include_faces=True)
        ranked_images = self.searcher.rank_images("A photo of a fast vehicle.")
        self.assertIsInstance(ranked_images, list)

        self.searcher = Search(image_dir_path=None, save_path=self.image_dir_path, include_faces=True)
        ranked_images = self.searcher.rank_images("A photo of a fast vehicle.")
        self.assertIsInstance(ranked_images, list)
        print(ranked_images)

    def test_face_searcher(self):
        self.searcher = Search(image_dir_path=self.image_dir_path, include_faces=True)
        ranked_images = self.searcher.rank_images_by_faces(image_path=os.path.join(self.image_dir_path, "profile.jpg"))
        self.assertIsInstance(ranked_images, list)

        self.searcher = Search(image_dir_path=None, save_path=self.image_dir_path, include_faces=True)
        ranked_images = self.searcher.rank_images_by_faces(image_path=os.path.join(self.image_dir_path, "friends.jpg"))
        self.assertIsInstance(ranked_images, list)
        print(ranked_images)

