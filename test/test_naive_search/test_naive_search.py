import unittest

from image_searcher.search.naive_search import NaiveSearch


class TestNaiveSearch(unittest.TestCase):
    def setUp(self):
        self.searcher = NaiveSearch(image_dir_path="/home/manu/perso/ImageSearcher/data/")

    def test_searcher(self):
        ranked_images = self.searcher.rank_images("A photo of a fast vehicle.")
        print(ranked_images)

        ranked_images = self.searcher.rank_images("A photo of a bird.")
        print(ranked_images)
