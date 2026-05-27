import unittest
import warnings

import numpy as np
import torch

from image_searcher.search.search import Search


class FakeStoredEmbeddings:
    def __init__(self):
        self.face_embedding_tensor = torch.from_numpy(np.asarray([
            np.zeros(128),
            np.ones(128),
            np.full(128, 2.0),
        ], dtype=np.float32))
        self.face_embedding_paths = ["zero.jpg", "one.jpg", "two.jpg"]
        self.embeddings = {
            "query.jpg": {"face_embeddings": [np.full(128, 1.1), np.full(128, 0.1)]},
            "no-face.jpg": {"face_embeddings": []},
        }

    def get_all_face_embeddings(self):
        return self.face_embedding_tensor, self.face_embedding_paths


class TestFaceRanking(unittest.TestCase):
    def setUp(self):
        self.searcher = Search.__new__(Search)
        self.searcher.stored_embeddings = FakeStoredEmbeddings()

    def test_face_ranking_is_vectorized_and_returns_closest_paths(self):
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            ranked_images = self.searcher.rank_images_by_faces("query.jpg", n=2)

        self.assertEqual([item.image_path for item in ranked_images[0]], ["one.jpg", "two.jpg"])
        self.assertEqual([item.image_path for item in ranked_images[1]], ["zero.jpg", "one.jpg"])
        self.assertFalse(any("Creating a tensor from a list" in str(w.message) for w in recorded_warnings))

    def test_no_detected_faces_returns_empty_results(self):
        self.assertEqual(self.searcher.rank_images_by_faces("no-face.jpg"), [])
