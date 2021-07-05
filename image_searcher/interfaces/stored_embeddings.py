import pickle
import os

import torch


class StoredEmbeddings:
    def __init__(self, save_path: str):
        self.save_path = save_path
        if os.path.isdir(save_path):
            self.save_path = os.path.join(save_path, "stored_embeddings.pickle")

        self.embeddings = {}
        if os.path.isfile(self.save_path):
            with open(self.save_path, "rb") as file:
                self.embeddings = pickle.load(file)

    def get_image_paths(self):
        return list(self.embeddings.keys())

    def add_embedding(self, image_path, embedding):
        assert embedding.shape == (1, 512)
        self.embeddings[image_path] = embedding

    def get_embedding_tensor(self):
        return torch.cat(tuple(self.embeddings.values()), dim=0)
        # paths, embeds = list(self.embeddings.keys()), tuple(self.embeddings.values())
        # return torch.cat(embeds, dim=0)

    def update_file(self):
        """This method should be called when reindex is done"""
        with open(self.save_path, "wb") as file:
            pickle.dump(self.embeddings, file)
