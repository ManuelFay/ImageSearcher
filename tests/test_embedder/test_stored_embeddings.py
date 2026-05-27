import tempfile
import unittest
import warnings

import numpy as np
import torch

from image_searcher.interfaces.stored_embeddings import StoredEmbeddings


class TestStoredEmbeddings(unittest.TestCase):
    def test_face_embeddings_are_combined_without_conversion_warning(self):
        with tempfile.TemporaryDirectory() as save_path:
            stored_embeddings = StoredEmbeddings(save_path)
            stored_embeddings.embeddings = {
                "one.jpg": {"face_embeddings": [np.ones(128), np.zeros(128)]},
                "two.jpg": {"face_embeddings": [np.full(128, 2.0)]},
            }

            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                stored_embeddings.set_all_face_embeddings()

            self.assertEqual(stored_embeddings.face_embedding_tensor.shape, torch.Size([3, 128]))
            self.assertEqual(stored_embeddings.face_embedding_tensor.dtype, torch.float32)
            self.assertEqual(stored_embeddings.face_embedding_paths, ["one.jpg", "one.jpg", "two.jpg"])
            self.assertFalse(any("Creating a tensor from a list" in str(w.message) for w in recorded_warnings))

    def test_no_faces_keeps_an_empty_tensor(self):
        with tempfile.TemporaryDirectory() as save_path:
            stored_embeddings = StoredEmbeddings(save_path)
            stored_embeddings.embeddings = {"one.jpg": {"face_embeddings": []}}

            stored_embeddings.set_all_face_embeddings()

            self.assertEqual(stored_embeddings.face_embedding_tensor.shape, torch.Size([0]))
